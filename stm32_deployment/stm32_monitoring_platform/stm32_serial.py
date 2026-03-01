import os
import time
import struct
import numpy as np
import serial
import serial.tools.list_ports
import pandas as pd
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import threading

# 全局超时配置
GLOBAL_TIMEOUT_SECONDS = 600


def make_deadline(seconds: float = GLOBAL_TIMEOUT_SECONDS) -> float:
    """生成基于当前时间的超时时间点"""
    return time.time() + seconds


def check_deadline(deadline: Optional[float], stage: str, tracker: "ClientStatsTracker" = None):
    """统一的全局超时检测"""
    if deadline and time.time() > deadline:
        if tracker:
            tracker.mark_error(get_status_text('global_timeout_stage').format(stage))
        raise TimeoutError(get_status_text('global_timeout_stage').format(stage))

# ============================
# 状态翻译函数（从session获取）
# ============================
def get_status_text(key: str) -> str:
    """获取翻译后的状态文本"""
    # 默认文案按语言分组，保证即使无法访问 Streamlit 也能返回对应语言
    default_texts = {
        'zh': {
            'status_waiting': '等待中',
            'status_preparing': '准备中',
            'status_training': '训练中',
            'status_uploading': '上传中',
            'status_sending': '发送数据中',
            'status_processing': '等待处理',
            'status_completed': '完成 ✓',
            'status_error': '错误 ✗',
            'status_retrying': '重试中',
            'status_waiting_retry': '准备重试',
            'wait_tag_timeout': '等待 {} 超时',
            'stm32_no_response': 'STM32 未响应',
            'stm32_no_ready': 'STM32 未响应 READY',
            'stm32_not_sending_ready': 'STM32 未发送 READY',
            'timeout_with_retries': '超时 (已重试{}次)',
            'global_timeout_stage': '全局超时 ({})',
        },
        'en': {
            'status_waiting': 'Waiting',
            'status_preparing': 'Preparing',
            'status_training': 'Training',
            'status_uploading': 'Uploading',
            'status_sending': 'Sending',
            'status_processing': 'Processing',
            'status_completed': 'Completed ✓',
            'status_error': 'Error ✗',
            'status_retrying': 'Retrying',
            'status_waiting_retry': 'Preparing retry',
            'wait_tag_timeout': 'Waiting for {} timed out',
            'stm32_no_response': 'STM32 no response',
            'stm32_no_ready': 'STM32 no READY',
            'stm32_not_sending_ready': 'STM32 did not send READY',
            'timeout_with_retries': 'Timeout (retried {} times)',
            'global_timeout_stage': 'Global timeout ({})',
        }
    }

    try:
        import streamlit as st
        lang = st.session_state.get('language', 'en')

        # 从 stm32_dashboard 导入语言字典
        from stm32_dashboard import LANGUAGES
        return LANGUAGES.get(lang, LANGUAGES['en']).get(key, key)
    except Exception:
        # 回退逻辑：默认英文，支持中文环境变量
        env_lang = os.environ.get('APP_LANGUAGE', '').lower()
        if env_lang.startswith('zh'):
            return default_texts['zh'].get(key, key)
        return default_texts['en'].get(key, key)

# ============================
# 串口配置（默认值）
# ============================
SERIAL_PORT = "COM4"
BAUDRATE = 460800
TIMEOUT = 5.0
SCALE = 100.0

# ============================
# SLIP 转义协议定义
# ============================
SLIP_END = 0xC0
SLIP_ESC = 0xDB
SLIP_ESC_END = 0xDC
SLIP_ESC_ESC = 0xDD
SLIP_ESC_LF = 0xDE   # 0x0A (换行符)
SLIP_ESC_CR = 0xDF   # 0x0D (回车符)



# ============================
# ★★★ SLIP 转义编码函数 ★★★
# ============================

def slip_encode(data: bytes) -> bytes:
    """
    对二进制数据进行 SLIP 转义编码
    
    参数:
        data: 原始字节流
    
    返回:
        bytes: 转义后的数据（包含帧头和帧尾 0xC0）
    """
    encoded = bytearray()
    
    # 帧开始标记
    encoded.append(SLIP_END)
    
    # 逐字节转义
    for byte in data:
        if byte == SLIP_END:
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_END)
        elif byte == SLIP_ESC:
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_ESC)
        elif byte == 0x0A:  # 换行符 LF
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_LF)
        elif byte == 0x0D:  # 回车符 CR
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_CR)
        else:
            encoded.append(byte)
    
    # 帧结束标记
    encoded.append(SLIP_END)
    
    return bytes(encoded)


def slip_encode_with_stats(data: bytes) -> tuple:
    """
    对二进制数据进行 SLIP 转义编码（带统计信息）
    
    返回:
        tuple: (encoded_data: bytes, stats: dict)
    """
    encoded = bytearray()
    
    # 统计信息
    stats = {
        'original_size': len(data),
        'escaped_LF': 0,
        'escaped_CR': 0,
        'escaped_END': 0,
        'escaped_ESC': 0,
        'total_escaped': 0
    }
    
    # 帧开始
    encoded.append(SLIP_END)
    
    # 转义
    for byte in data:
        if byte == SLIP_END:
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_END)
            stats['escaped_END'] += 1
            stats['total_escaped'] += 1
        elif byte == SLIP_ESC:
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_ESC)
            stats['escaped_ESC'] += 1
            stats['total_escaped'] += 1
        elif byte == 0x0A:
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_LF)
            stats['escaped_LF'] += 1
            stats['total_escaped'] += 1
        elif byte == 0x0D:
            encoded.append(SLIP_ESC)
            encoded.append(SLIP_ESC_CR)
            stats['escaped_CR'] += 1
            stats['total_escaped'] += 1
        else:
            encoded.append(byte)
    
    # 帧结束
    encoded.append(SLIP_END)
    
    stats['encoded_size'] = len(encoded)
    stats['overhead'] = len(encoded) - len(data)
    stats['overhead_percent'] = (stats['overhead'] / len(data)) * 100 if len(data) > 0 else 0
    
    return bytes(encoded), stats


# ============================
# 状态数据类（供UI使用）
# ============================

@dataclass
class ClientStats:
    """单个客户端的统计数据"""
    client_id: int
    status: str = ""  # 将在初始化后设置为翻译后的文本
    progress: float = 0.0
    training_time: Optional[float] = None
    upload_time: Optional[float] = None
    upload_bytes: int = 0
    error_msg: Optional[str] = None
    retry_count: int = 0  # ★★★ 新增：记录重试次数 ★★★
    
    def __post_init__(self):
        if not self.status:
            self.status = get_status_text('status_waiting')


@dataclass
class AppState:
    """应用全局状态"""
    is_running: bool = False
    is_paused: bool = False
    current_client: int = -1
    total_clients: int = 0
    clients: List[ClientStats] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    serial_connected: bool = False
    
    # 双设备状态（★ 新增）
    device1_current_client: int = -1
    device2_current_client: int = -1
    
    # 配置
    serial_port: str = "COM3"
    baudrate: int = 460800
    scale: float = 100.0


# ============================
# 统计跟踪器
# ============================

class ClientStatsTracker:
    """统计每个客户端的训练时间、上传时间、数据量"""

    TRAINING_START_KEYWORD = "Each sample:"
    TRAINING_END_KEYWORD = "Starting X^T @ X"

    def __init__(self, app_state: AppState = None):
        self.app_state = app_state
        self.records = []
        
        self.current_client_id = None
        self.training_start_time = None
        self.training_end_time = None
        self.upload_end_time = None
        self.upload_bytes = 0

    def log(self, message: str):
        """记录日志"""
        print(message)
        if self.app_state:
            timestamp = time.strftime('%H:%M:%S')
            self.app_state.logs.append(f"[{timestamp}] {message}")
            if len(self.app_state.logs) > 500:
                self.app_state.logs = self.app_state.logs[-300:]

    def update_client_status(self, status: str, progress: float = None):
        """更新当前客户端状态"""
        if self.app_state and self.current_client_id is not None:
            if self.current_client_id < len(self.app_state.clients):
                client = self.app_state.clients[self.current_client_id]
                client.status = status
                if progress is not None:
                    client.progress = progress

    def start_client(self, client_id: int):
        """开始跟踪一个新客户端"""
        self.current_client_id = client_id
        self.training_start_time = None
        self.training_end_time = None
        self.upload_end_time = None
        self.upload_bytes = 0
        
        self.log(f"[Stats] 开始跟踪客户端 {client_id}")
        self.update_client_status(get_status_text('status_preparing'), 5)
        
        # ★★★ 新增：清除之前的错误信息 ★★★
        if self.app_state and client_id < len(self.app_state.clients):
            self.app_state.clients[client_id].error_msg = None
        
        if self.app_state:
            self.app_state.current_client = client_id

    def check_and_record(self, text: str):
        """检查接收到的文本，识别关键词并记录时间戳"""
        if not text:
            return None

        if self.TRAINING_START_KEYWORD in text:
            self.training_start_time = time.time()
            self.log(f"[Stats] ⏱ 检测到训练开始")
            self.update_client_status(get_status_text('status_training'), 30)
            return 'training_start'

        if self.TRAINING_END_KEYWORD in text:
            self.training_end_time = time.time()
            if self.training_start_time:
                elapsed = self.training_end_time - self.training_start_time
                self.log(f"[Stats] ⏱ 训练完成，耗时: {elapsed:.2f}s")
                if self.app_state and self.current_client_id < len(self.app_state.clients):
                    self.app_state.clients[self.current_client_id].training_time = elapsed
            
            self.log(f"[Stats] ⏱ 开始上传数据...")
            self.update_client_status(get_status_text('status_uploading'), 50)
            return 'training_end'

        return None

    def add_binary_data(self, byte_count: int):
        """累加接收到的二进制字节数"""
        self.upload_bytes += byte_count
        
        if self.app_state and self.current_client_id is not None:
            if self.current_client_id < len(self.app_state.clients):
                self.app_state.clients[self.current_client_id].upload_bytes = self.upload_bytes

    def update_upload_progress(self, progress: float):
        """更新上传进度（0-100）"""
        mapped_progress = 50 + progress * 0.45
        self.update_client_status(get_status_text('status_uploading'), mapped_progress)

    def finish_upload(self):
        """标记上传结束"""
        self.upload_end_time = time.time()
        if self.training_end_time:
            elapsed = self.upload_end_time - self.training_end_time
            self.log(f"[Stats] ⏱ 上传完成，耗时: {elapsed:.2f}s")
            self.log(f"[Stats] 📦 数据量: {self.upload_bytes:,} 字节 ({self.upload_bytes / 1024:.2f} KB)")
            
            if self.app_state and self.current_client_id < len(self.app_state.clients):
                self.app_state.clients[self.current_client_id].upload_time = elapsed

    def finish_client(self):
        """完成当前客户端，保存统计记录"""
        training_time = None
        upload_time = None

        if self.training_start_time and self.training_end_time:
            training_time = self.training_end_time - self.training_start_time

        if self.training_end_time and self.upload_end_time:
            upload_time = self.upload_end_time - self.training_end_time

        record = {
            'client_id': self.current_client_id,
            'training_time_s': round(training_time, 3) if training_time else None,
            'upload_time_s': round(upload_time, 3) if upload_time else None,
            'upload_bytes': self.upload_bytes,
            'upload_KB': round(self.upload_bytes / 1024, 2) if self.upload_bytes else 0,
            'upload_MB': round(self.upload_bytes / 1024 / 1024, 4) if self.upload_bytes else 0,
        }

        self.records.append(record)
        self.update_client_status(get_status_text('status_completed'), 100)

        self.log(f"[Stats] ══════ 客户端 {self.current_client_id} 完成 ══════")

    def mark_error(self, error_msg: str):
        """标记客户端错误"""
        self.update_client_status(get_status_text('status_error'), 0)
        if self.app_state and self.current_client_id is not None:
            if self.current_client_id < len(self.app_state.clients):
                self.app_state.clients[self.current_client_id].error_msg = error_msg

    def get_dataframe(self) -> pd.DataFrame:
        """获取统计数据的 DataFrame"""
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

    def save_to_excel(self, filepath: str = "client_stats.xlsx"):
        """保存统计数据到Excel文件"""
        if not self.records:
            self.log("[Stats] 没有统计数据可保存")
            return

        df = pd.DataFrame(self.records)
        
        summary_row = {
            'client_id': 'TOTAL/AVG',
            'training_time_s': df['training_time_s'].mean(),
            'upload_time_s': df['upload_time_s'].mean(),
            'upload_bytes': df['upload_bytes'].sum(),
            'upload_KB': df['upload_KB'].sum(),
            'upload_MB': df['upload_MB'].sum(),
        }

        df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_with_summary.to_excel(writer, sheet_name='Client Stats', index=False)

        self.log(f"[Stats] ✓ 统计数据已保存到 {filepath}")

    def clear_error(self):
        """清除当前客户端的错误信息"""
        if self.app_state and self.current_client_id is not None:
            if self.current_client_id < len(self.app_state.clients):
                self.app_state.clients[self.current_client_id].error_msg = None



# ============================
# 串口辅助函数
# ============================

def get_available_ports() -> list:
    """获取可用的串口列表"""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def open_serial(port: str = SERIAL_PORT, baudrate: int = BAUDRATE):
    """打开串口"""
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=TIMEOUT)
    time.sleep(0.5)
    return ser


def wait_for_response(ser, keyword, tracker: ClientStatsTracker, deadline: Optional[float]):
    """等待串口接收到包含指定关键词的行（使用局部超时，不依赖全局deadline）"""
    tracker.log(f"[Python] 等待响应: '{keyword}'")

    timeout = 15.0
    if deadline:
        timeout = max(1.0, deadline - time.time())

    start_time = time.time()
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            try:
                line = ser.readline()
                if not line:
                    continue
                text = line.decode('utf-8', errors='ignore').strip()
                
                if text and all(ord(c) < 128 and (c.isprintable() or c.isspace()) for c in text):
                    tracker.log(f"[RX] {text}")
                    tracker.check_and_record(text)
                    
                    if keyword.lower() in text.lower():
                        tracker.log(f"[Python] ✓ 收到 '{keyword}'")
                        return True
            except Exception as e:
                tracker.log(f"[Python] 读取异常: {e}")
                continue
        time.sleep(0.01)

    error_msg = f"等待 '{keyword}' 超时 ({timeout:.1f}s)"
    tracker.log(f"[Python] ✗ {error_msg}")
    tracker.mark_error(error_msg)
    raise TimeoutError(error_msg)


# ============================
# 数据加载函数
# ============================

def load_client_data_from_excel(data_dir: str = "./output/client_data"):
    """从 Excel 文件加载客户端数据"""
    import torch
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"文件夹 '{data_dir}' 不存在！")

    client_files = sorted(glob.glob(os.path.join(data_dir, "client_*_data.xlsx")))

    if len(client_files) == 0:
        raise FileNotFoundError(f"在 '{data_dir}' 中没有找到 client_*_data.xlsx 文件！")

    train_dataset = []
    
    for file_path in client_files:
        df = pd.read_excel(file_path)
        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        train_dataset.append(data_tensor)

    return train_dataset


def prepare_client_data_int16(dataset, clip_min=-100.0, clip_max=100.0, scale=SCALE):
    """将数据转换为 int16 格式"""
    data = dataset.numpy()
    data = data[:, -151:]
    np.clip(data, clip_min, clip_max, out=data)
    data = np.round(data, 2)
    data_int16 = (data * scale).astype(np.int16)
    return data_int16


# ============================
# ★★★ SLIP 转义解码函数（带进度条）★★★
# ============================

def recv_escaped_binary(ser, tracker: ClientStatsTracker, deadline: Optional[float]):
    """接收 SLIP 转义的二进制数据（局部超时：整体+空闲）"""
    data = bytearray()
    escape_next = False
    frame_started = False
    
    # 用于定期更新进度
    last_update_bytes = 0
    UPDATE_INTERVAL_BYTES = 4096

    # 总体超时与空闲超时
    overall_timeout = 300.0
    if deadline:
        overall_timeout = max(1.0, deadline - time.time())
    idle_timeout = 10.0
    start_time = time.time()
    last_data_time = start_time

    while time.time() - start_time < overall_timeout:
        byte_data = ser.read(1)
        if not byte_data:
            if frame_started and time.time() - last_data_time > idle_timeout:
                error_msg = f"SLIP接收空闲超时：{idle_timeout}秒内未收到数据（已接收{len(data)}字节）"
                tracker.log(f"[ERROR] {error_msg}")
                raise TimeoutError(error_msg)
            continue
        
        last_data_time = time.time()
        byte = byte_data[0]

        # 帧边界检测
        if byte == SLIP_END:
            if frame_started:
                remaining = len(data) - last_update_bytes
                if remaining > 0:
                    tracker.add_binary_data(remaining)
                return bytes(data)
            else:
                frame_started = True
                data.clear()
                last_update_bytes = 0
                continue

        if not frame_started:
            continue

        # 转义处理
        if escape_next:
            if byte == SLIP_ESC_END:
                data.append(SLIP_END)
            elif byte == SLIP_ESC_ESC:
                data.append(SLIP_ESC)
            elif byte == SLIP_ESC_LF:
                data.append(0x0A)
            elif byte == SLIP_ESC_CR:
                data.append(0x0D)
            else:
                data.append(byte)
            escape_next = False
        elif byte == SLIP_ESC:
            escape_next = True
        else:
            data.append(byte)
        
        # 定期更新数据量
        current_len = len(data)
        if current_len - last_update_bytes >= UPDATE_INTERVAL_BYTES:
            bytes_to_add = current_len - last_update_bytes
            tracker.add_binary_data(bytes_to_add)
            last_update_bytes = current_len

    error_msg = f"SLIP 接收超时（{overall_timeout:.1f}s），已接收 {len(data)} 字节"
    tracker.log(f"[ERROR] {error_msg}")
    raise TimeoutError(error_msg)




# ============================
# ★★★ 接收二进制格式的矩阵（SLIP 转义）★★★
# ============================

def recv_matrix_binary(ser, tracker: ClientStatsTracker, tag="W_SHAPE", deadline: Optional[float] = None):
    """接收二进制格式的矩阵（局部超时）"""
    tracker.log(f"[Python] 等待接收 {tag} ...")

    timeout = 600.0
    if deadline:
        timeout = max(1.0, deadline - time.time())

    start_time = time.time()

    # 等待头部
    header = None
    while time.time() - start_time < timeout:
        line = ser.readline()
        if not line:
            time.sleep(0.01)
            continue

        try:
            text = line.decode(errors="ignore").strip()
        except:
            continue

        if text and len(text) < 200:
            tracker.log(f"[RX] {text}")
            tracker.check_and_record(text)

        if text.startswith(tag):
            header = text
            break

    if header is None:
        raise TimeoutError(get_status_text('wait_tag_timeout').format(tag))

    # 解析头部
    parts = header.split()
    if len(parts) < 4 or parts[3] != "BIN":
        raise ValueError(f"{tag} 头部格式错误: {header!r}")

    rows = int(parts[1])
    cols = int(parts[2])

    tracker.log(f"[Python] 接收 {tag}: [{rows} x {cols}] (二进制)")

    # 接收 SLIP 数据
    raw_data = recv_escaped_binary(ser, tracker, None)

    # 验证并解析
    expected_bytes = rows * cols * 4
    if len(raw_data) != expected_bytes:
        tracker.log(f"[WARNING] {tag} 数据长度不匹配: 期望{expected_bytes}, 实际{len(raw_data)}")

    try:
        matrix = np.frombuffer(raw_data, dtype='<f4').reshape(rows, cols)

        tracker.log(f"[Python] ✓ 成功接收 {tag}")
        tracker.log(f"         形状: {matrix.shape}, 范围: [{matrix.min():.4f}, {matrix.max():.4f}]")

        tracker.update_upload_progress(100)

        # 等待 END
        for _ in range(10):
            line = ser.readline()
            if line:
                text = line.decode(errors='ignore').strip()
                tracker.log(f"[RX] {text}")
                if text == "END":
                    break

        return matrix

    except Exception as e:
        tracker.log(f"[ERROR] {tag} 解析失败: {e}")
        raise



# ============================
# ★★★ 接收 X^T @ X 矩阵（二进制流式 + SLIP）★★★
# ============================

def recv_xtx_streaming_binary(ser, tracker: ClientStatsTracker, deadline: Optional[float]):
    """接收 STM32 分块发送的 X^T @ X 矩阵（局部超时）"""
    tracker.log(f"\n[Python] 开始接收 X^T @ X ...")

    overall_timeout = 600.0
    if deadline:
        overall_timeout = max(1.0, deadline - time.time())
    start_time = time.time()

    # 等待 XTX_START
    matrix_size = None
    total_blocks = None

    while time.time() - start_time < overall_timeout:
        try:
            line = ser.readline()
            if not line:
                continue
            
            text = line.decode(errors='ignore').strip()

            if text.startswith("XTX_START"):
                parts = text.split()
                if len(parts) >= 4 and parts[3] == "BIN":
                    matrix_size = int(parts[1])
                    total_blocks = int(parts[2])
                    tracker.log(f"[Python] X^T @ X: [{matrix_size} x {matrix_size}], {total_blocks} 块")
                    break
            elif text and len(text) < 200:
                tracker.log(f"[RX] {text[:100]}")
                tracker.check_and_record(text)
        except:
            continue

    if matrix_size is None:
        tracker.log("[ERROR] 未收到 XTX_START")
        raise TimeoutError("未收到 XTX_START")

    # 初始化矩阵
    XTX = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    blocks_received = 0

    # 接收每个块
    while blocks_received < total_blocks:
        # 等待块头
        header_found = False
        row_start = col_start = block_rows = block_cols = 0

        while time.time() - start_time < overall_timeout:
            line = ser.readline()
            if not line:
                time.sleep(0.01)
                continue
            
            try:
                text = line.decode(errors='ignore').strip()
            except:
                continue

            if text.startswith("XTX_BLOCK"):
                parts = text.split()
                if len(parts) >= 6 and parts[5] == "BIN":
                    row_start = int(parts[1])
                    col_start = int(parts[2])
                    block_rows = int(parts[3])
                    block_cols = int(parts[4])
                    header_found = True
                    tracker.log(f"\n[RX] 块 {blocks_received + 1}/{total_blocks}: "
                              f"[{row_start}:{row_start + block_rows}, "
                              f"{col_start}:{col_start + block_cols}]")
                    break
            elif text and len(text) < 200:
                tracker.log(f"[STM32] {text}")
                tracker.check_and_record(text)

        if not header_found:
            tracker.log(f"[ERROR] 未收到块 {blocks_received + 1} 的头")
            raise TimeoutError(f"未收到块 {blocks_received + 1} 的头")

        # 接收 SLIP 数据（带空闲检测）
        block_data_raw = recv_escaped_binary(
            ser, tracker,
            None
        )

        # 解析并存储
        expected_bytes = block_rows * block_cols * 4
        
        try:
            if len(block_data_raw) != expected_bytes:
                tracker.log(f"[WARNING] 块数据长度不匹配: 期望{expected_bytes}, 实际{len(block_data_raw)}")

            block_data = np.frombuffer(block_data_raw, dtype='<f4').reshape(block_rows, block_cols)

            XTX[row_start:row_start + block_rows, col_start:col_start + block_cols] = block_data

            if row_start != col_start:
                XTX[col_start:col_start + block_cols, row_start:row_start + block_rows] = block_data.T

            blocks_received += 1

            progress = (blocks_received / total_blocks) * 100
            tracker.update_upload_progress(progress)
            
            tracker.log(f"[Python] ✓ 块 {blocks_received}/{total_blocks} 完成 ({progress:.1f}%)")

        except Exception as e:
            tracker.log(f"[ERROR] 块解析失败: {e}")
            raise

        # 等待 BLOCK_END
        for _ in range(5):
            line = ser.readline()
            if line:
                text = line.decode(errors='ignore').strip()
                if text:
                    tracker.log(f"[RX] {text}")
                if "BLOCK_END" in text:
                    tracker.log("[Python] ✓ 收到 BLOCK_END")
                    break
            line = ser.readline()
            if line:
                text = line.decode(errors='ignore').strip()
                if text == "BLOCK_END":
                    break

    # 等待 XTX_END
    for _ in range(50):
        check_deadline(deadline, "等待 XTX_END", tracker)
        try:
            line = ser.readline()
            if line:
                if "XTX_END" in line.decode(errors='ignore'):
                    tracker.log("[Python] ✓ 收到 XTX_END")
                    break
        except:
            pass

    tracker.log(f"[Python] ✓ X^T @ X 接收完成, 形状: {XTX.shape}")
    return XTX



# ============================
# 发送客户端数据
# ============================

def send_client_data(ser, client_id, data_int16, tracker: ClientStatsTracker, app_state: AppState, deadline: Optional[float]):
    """发送一个客户端的数据到 STM32（使用 SLIP 转义，统一全局超时）"""
    num_samples, num_features = data_int16.shape

    tracker.log(f"\n[Python] 准备发送客户端 {client_id} 数据：{num_samples} x {num_features}")
    tracker.start_client(client_id)

    # 发送命令
    ser.write(f"CLIENT_BEGIN {client_id}\n".encode())
    ser.flush()
    time.sleep(0.1)

    ser.write(f"SCALE {app_state.scale}\n".encode())
    ser.flush()
    time.sleep(0.1)

    ser.write(f"SAMPLES {num_samples}\n".encode())
    ser.flush()
    time.sleep(0.1)

    ser.write(f"FEATURES {num_features}\n".encode())
    ser.flush()
    time.sleep(0.2)

    tracker.update_client_status(get_status_text('status_sending'), 10)

    # 等待 STM32 准备
    check_deadline(deadline, "等待设备就绪", tracker)
    if not wait_for_response(ser, "ready to receive binary data", tracker, deadline):
        tracker.mark_error(get_status_text('stm32_no_response'))
        raise TimeoutError(get_status_text('stm32_no_response'))

    tracker.log(f"[Python] STM32 已准备好，开始发送...")
    tracker.update_client_status(get_status_text('status_sending'), 15)

    # ★★★ 修改：使用 SLIP 转义编码 ★★★
    raw = data_int16.astype('<i2', copy=False).tobytes(order='C')
    
    # SLIP 转义
    encoded_data, stats = slip_encode_with_stats(raw)
    
    # 打印统计信息
    tracker.log(f"\n[Python] ========== SLIP 转义统计 ==========")
    tracker.log(f"         原始大小:   {stats['original_size']} 字节")
    tracker.log(f"         转义后:     {stats['encoded_size']} 字节")
    tracker.log(f"         开销:       {stats['overhead']} 字节 ({stats['overhead_percent']:.2f}%)")
    tracker.log(f"         转义字符:")
    tracker.log(f"           换行符 (0x0A): {stats['escaped_LF']} 次")
    tracker.log(f"           回车符 (0x0D): {stats['escaped_CR']} 次")
    tracker.log(f"           帧边界 (0xC0): {stats['escaped_END']} 次")
    tracker.log(f"           转义符 (0xDB): {stats['escaped_ESC']} 次")
    tracker.log(f"           总计:          {stats['total_escaped']} 个字节")
    tracker.log(f"=========================================\n")

    CHUNK_SIZE = 1024
    total_sent = 0
    start_time = time.time()

    # ★★★ 发送转义后的数据 ★★★
    while total_sent < len(encoded_data):
        check_deadline(deadline, "发送数据", tracker)
        chunk = encoded_data[total_sent:total_sent + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        total_sent += len(chunk)

        if total_sent % 4096 == 0 or total_sent == len(encoded_data):
            progress = (total_sent / len(encoded_data)) * 100
            tracker.log(f"[TX] 进度: {total_sent}/{len(encoded_data)} ({progress:.1f}%)")
            mapped_progress = 10 + (progress / 100) * 15
            tracker.update_client_status(get_status_text('status_sending'), mapped_progress)

        time.sleep(0.05)  # ★ 恢复快速发送（SLIP 解决了 MSH 问题）

    elapsed = time.time() - start_time
    speed = len(encoded_data) / elapsed / 1024 if elapsed > 0 else 0
    tracker.log(f"\n[TX] 传输完成: {elapsed:.2f}s, 速度: {speed:.2f} KB/s")
    tracker.update_client_status(get_status_text('status_processing'), 25)

    # 等待 READY
    tracker.log(f"[Python] 等待 STM32 处理完成...")

    check_deadline(deadline, "等待 READY", tracker)
    if not wait_for_response(ser, "READY", tracker, deadline):
        tracker.mark_error(get_status_text('stm32_not_sending_ready'))
        raise TimeoutError(get_status_text('stm32_no_ready'))

    tracker.log(f"[Python] ✓ 收到 READY")

    # 发送结束标志
    time.sleep(0.2)
    ser.write(b"CLIENT_END\n")
    ser.flush()

    tracker.log("[Python] 等待设备计算结果...")

    # 标记进入结果接收阶段，避免仍显示为训练中
    tracker.update_client_status(get_status_text('status_uploading'), 60)

    # 接收结果
    check_deadline(deadline, "接收 W", tracker)
    W = recv_matrix_binary(ser, tracker, tag="W_SHAPE", deadline=deadline)
    check_deadline(deadline, "接收 XTX", tracker)
    XTX = recv_xtx_streaming_binary(ser, tracker, deadline=deadline)

    tracker.finish_upload()
    tracker.finish_client()
    time.sleep(1)

    tracker.log(f"[Python] ✓ 客户端 {client_id} 完成")
    tracker.log(f"         W: {W.shape}, XTX: {XTX.shape}")

    return W, XTX



# ============================
# ★★★ 新增：自动保存功能 ★★★
# ============================

def save_checkpoint(
    all_W: list,
    all_XTXs: list,
    tracker: ClientStatsTracker,
    save_dir: str = "./output",
    reason: str = "checkpoint"
) -> tuple:
    """
    保存当前已完成的结果（用于中断恢复）
    
    返回: (success: bool, message: str)
    """
    from datetime import datetime
    
    if not all_W:
        return False, "没有可保存的数据"
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存模型结果
        result_path = os.path.join(save_dir, f"checkpoint_{timestamp}.npz")
        np.savez(result_path, W=all_W, XTX=all_XTXs)
        tracker.log(f"[AutoSave] ✓ 模型已保存: {result_path} ({len(all_W)} 个客户端)")
        
        # 2. 保存统计数据
        stats_path = os.path.join(save_dir, f"stats_{timestamp}.xlsx")
        tracker.save_to_excel(stats_path)
        
        # 3. 保存日志
        if tracker.app_state and tracker.app_state.logs:
            log_path = os.path.join(save_dir, f"logs_{timestamp}.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"# 保存原因: {reason}\n")
                f.write(f"# 保存时间: {datetime.now().isoformat()}\n")
                f.write(f"# 完成客户端: {len(all_W)}/{tracker.app_state.total_clients}\n")
                f.write("=" * 60 + "\n\n")
                for log in tracker.app_state.logs:
                    f.write(log + "\n")
            tracker.log(f"[AutoSave] ✓ 日志已保存: {log_path}")
        
        return True, f"已保存到 {save_dir}"
        
    except Exception as e:
        error_msg = f"保存失败: {e}"
        tracker.log(f"[AutoSave] ✗ {error_msg}")
        return False, error_msg


# ============================
# ★★★ 主流程（带状态管理）★★★
# ============================

def run_all_clients_with_autosave(
    train_dataset,
    app_state: AppState,
    tracker: ClientStatsTracker,
    stop_event: threading.Event,
    save_path: str = "./output/results.npz",
    stats_excel_path: str = "./output/client_stats.xlsx",
    checkpoint_interval: int = 5,  # 每完成N个客户端自动保存
    global_timeout_seconds: float = GLOBAL_TIMEOUT_SECONDS
):
    """
    ★★★ 带自动保存的主流程（封装原有函数）★★★
    
    - 中断时自动保存已完成的结果
    - 每完成 checkpoint_interval 个客户端自动保存检查点
    - 异常时自动保存
    """
    all_W = []
    all_XTXs = []
    last_data_int16 = None
    last_checkpoint_count = 0
    output_dir = os.path.dirname(save_path) or "./output"

    # 打开串口
    try:
        ser = open_serial(app_state.serial_port, app_state.baudrate)
        app_state.serial_connected = True
        tracker.log(f"[Python] ✓ 串口已打开: {app_state.serial_port} @ {app_state.baudrate}")
    except Exception as e:
        tracker.log(f"[Error] 无法打开串口: {e}")
        app_state.serial_connected = False
        app_state.is_running = False
        raise

    try:
        for client_id, ds in enumerate(train_dataset):
            # 检查停止信号
            if stop_event.is_set():
                tracker.log("[Python] 收到停止信号，正在保存...")
                # ★★★ 停止时自动保存 ★★★
                save_checkpoint(all_W, all_XTXs, tracker, output_dir, "用户停止")
                break

            # 检查暂停
            while app_state.is_paused:
                if stop_event.is_set():
                    break
                time.sleep(0.1)

            tracker.log(f"\n{'=' * 60}")
            tracker.log(f"开始处理客户端 {client_id}")
            tracker.log(f"{'=' * 60}")

            app_state.current_client = client_id

            # 准备数据
            data_int16 = prepare_client_data_int16(ds, scale=app_state.scale)
            tracker.log(f"[Python] 数据准备完毕: {data_int16.shape}")

            # 发送并接收
            try:
                deadline = make_deadline(global_timeout_seconds)
                W, XTX = send_client_data(ser, client_id, data_int16, tracker, app_state, deadline)

                all_W.append(W)
                all_XTXs.append(XTX)
                last_data_int16 = data_int16

                # ★★★ 增量自动保存 ★★★
                if len(all_W) > 0 and len(all_W) % checkpoint_interval == 0 and len(all_W) > last_checkpoint_count:
                    tracker.log(f"[AutoSave] 已完成 {len(all_W)} 个客户端，保存检查点...")
                    save_checkpoint(all_W, all_XTXs, tracker, output_dir, f"检查点 ({len(all_W)} clients)")
                    last_checkpoint_count = len(all_W)

                tracker.log(f"[Python] 休息 10s 防过热...")
                time.sleep(10.0)

            except TimeoutError as e:
                tracker.log(f"[Error] 客户端 {client_id} 超时: {e}")
                tracker.mark_error(str(e))
                # ★★★ 超时时也保存 ★★★
                if all_W:
                    save_checkpoint(all_W, all_XTXs, tracker, output_dir, f"超时中断 (client {client_id})")
                
                tracker.log(f"[Python] 超时冷却 10s...")
                time.sleep(10.0)
                continue

            except Exception as e:
                tracker.log(f"[Error] 客户端 {client_id} 错误: {e}")
                tracker.mark_error(str(e))
                # ★★★ 异常时保存 ★★★
                if all_W:
                    save_checkpoint(all_W, all_XTXs, tracker, output_dir, f"异常中断: {e}")
                raise

        # 正常完成 - 保存最终结果
        tracker.log(f"\n{'=' * 60}")
        tracker.log(f"所有客户端处理完成，成功: {len(all_W)}")
        tracker.log(f"{'=' * 60}")

        # 确保输出目录存在
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if save_path and all_W:
            save_file = save_path if save_path.endswith('.npz') else f"{save_path}_results.npz"
            np.savez(save_file, W=all_W, XTX=all_XTXs, last_data=last_data_int16)
            tracker.log(f"[Python] ✓ 最终结果已保存: {save_file}")

        # 保存统计
        tracker.save_to_excel(stats_excel_path)

    except KeyboardInterrupt:
        tracker.log("\n[Python] 用户中断 (Ctrl+C)")
        # ★★★ Ctrl+C 中断时保存 ★★★
        if all_W:
            save_checkpoint(all_W, all_XTXs, tracker, output_dir, "Ctrl+C 中断")
        tracker.save_to_excel(stats_excel_path)

    except Exception as e:
        tracker.log(f"\n[Python] 错误: {e}")
        # ★★★ 异常时保存 ★★★
        if all_W:
            save_checkpoint(all_W, all_XTXs, tracker, output_dir, f"异常: {e}")
        tracker.save_to_excel(stats_excel_path)
        raise

    finally:
        app_state.is_running = False
        ser.close()
        app_state.serial_connected = False
        tracker.log("[Python] ✓ 串口已关闭")

    return all_W, all_XTXs, last_data_int16


# ============================
# ★★★ 模拟测试（无需真实硬件）★★★
# ============================

def run_simulation(
    num_clients: int,
    app_state: AppState,
    tracker: ClientStatsTracker,
    stop_event: threading.Event
):
    """
    模拟运行（用于测试 UI，无需真实串口）
    """
    import random

    tracker.log("[Simulation] 开始模拟测试...")
    tracker.log(f"[Simulation] 模拟 {num_clients} 个客户端")

    for client_id in range(num_clients):
        # 检查停止
        if stop_event.is_set():
            tracker.log("[Simulation] 收到停止信号")
            break

        # 检查暂停
        while app_state.is_paused:
            if stop_event.is_set():
                break
            time.sleep(0.1)

        tracker.log(f"\n{'=' * 50}")
        tracker.log(f"[Simulation] 处理客户端 {client_id}")
        tracker.log(f"{'=' * 50}")

        tracker.start_client(client_id)
        app_state.current_client = client_id

        # 模拟发送数据
        tracker.update_client_status(get_status_text('status_sending'), 10)
        for i in range(5):
            if stop_event.is_set():
                break
            progress = 10 + i * 3
            tracker.update_client_status(get_status_text('status_sending'), progress)
            tracker.log(f"[Simulation] 发送进度: {i+1}/5")
            time.sleep(0.2)

        # 模拟训练
        tracker.training_start_time = time.time()
        tracker.update_client_status(get_status_text('status_training'), 30)
        tracker.log("[Simulation] STM32 训练中...")
        
        # 模拟训练时间 (1-3秒)
        train_time = random.uniform(1.0, 3.0)
        steps = 10
        for i in range(steps):
            if stop_event.is_set():
                break
            progress = 30 + (i / steps) * 20
            tracker.update_client_status(get_status_text('status_training'), progress)
            time.sleep(train_time / steps)

        tracker.training_end_time = time.time()
        if tracker.app_state and client_id < len(tracker.app_state.clients):
            tracker.app_state.clients[client_id].training_time = train_time

        tracker.log(f"[Simulation] 训练完成: {train_time:.2f}s")

        # 模拟上传
        tracker.update_client_status(get_status_text('status_uploading'), 50)
        tracker.log("[Simulation] 上传数据...")
        
        # 模拟上传时间 (0.5-2秒)
        upload_time = random.uniform(0.5, 2.0)
        upload_bytes = random.randint(50000, 200000)
        
        steps = 10
        for i in range(steps):
            if stop_event.is_set():
                break
            progress = 50 + (i / steps) * 45
            tracker.update_client_status(get_status_text('status_uploading'), progress)
            tracker.add_binary_data(upload_bytes // steps)
            time.sleep(upload_time / steps)

        tracker.upload_end_time = time.time()
        if tracker.app_state and client_id < len(tracker.app_state.clients):
            tracker.app_state.clients[client_id].upload_time = upload_time

        tracker.log(f"[Simulation] 上传完成: {upload_time:.2f}s, {upload_bytes} 字节")

        # 完成
        tracker.finish_client()
        tracker.log(f"[Simulation] ✓ 客户端 {client_id} 完成")

        time.sleep(2.0)

    tracker.log("\n" + "=" * 50)
    tracker.log("[Simulation] ✓ 模拟测试完成")
    tracker.log("=" * 50)

    # 保存统计
    tracker.save_to_excel("./output/simulation_stats.xlsx")

    app_state.is_running = False


# ============================
# 测试函数
# ============================

def test_serial_connection(port: str = "COM3", baudrate: int = 460800) -> dict:
    """测试串口连接"""
    result = {'success': False, 'message': ''}

    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=2.0)
        time.sleep(0.5)

        ser.write(b"TEST\n")
        ser.flush()

        start_time = time.time()
        while time.time() - start_time < 2.0:
            if ser.in_waiting > 0:
                data = ser.readline()
                if data:
                    result['success'] = True
                    response = data.decode(errors='ignore').strip()
                    result['message'] = get_status_text('serial_test_success').format(response)
                    break
            time.sleep(0.1)

        if not result['success']:
            result['success'] = True
            result['message'] = get_status_text('serial_test_opened').format(port)

        ser.close()

    except serial.SerialException as e:
        result['message'] = get_status_text('serial_test_error').format(e)

    except Exception as e:
        result['message'] = get_status_text('serial_test_unknown_error').format(e)

    return result


# ============================
# 独立运行测试
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("STM32 串口通信模块 (二进制 + SLIP)")
    print("=" * 60)

    # 检测串口
    ports = get_available_ports()
    print(f"\n可用串口: {ports}")

    if ports:
        result = test_serial_connection(ports[0])
        print(f"测试结果: {result['message']}")

    # 简单模拟测试
    print("\n运行模拟测试...")
    
    app_state = AppState()
    app_state.total_clients = 3
    app_state.clients = [ClientStats(client_id=i) for i in range(3)]
    app_state.is_running = True
    
    tracker = ClientStatsTracker(app_state)
    stop_event = threading.Event()
    
    run_simulation(3, app_state, tracker, stop_event)
    
    print("\n统计数据:")
    print(tracker.get_dataframe())

# ============================
# ★★★ 新增：双设备并行处理 ★★★
# ============================

@dataclass
@dataclass
class DualDeviceState:
    """双设备状态管理"""
    # 设备1配置（处理偶数客户端：0, 2, 4, ...）
    device1_port: str = "COM3"
    device1_baudrate: int = 460800
    device1_connected: bool = False
    device1_current_client: int = -1  # ★ 已有
    
    # 设备2配置（处理奇数客户端：1, 3, 5, ...）
    device2_port: str = "COM4"
    device2_baudrate: int = 460800
    device2_connected: bool = False
    device2_current_client: int = -1  # ★ 已有
    
    # 共享配置
    scale: float = 100.0


class DualDeviceTracker:
    """双设备统计跟踪器（包装两个独立的 ClientStatsTracker）"""
    
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.tracker1 = ClientStatsTracker(app_state)  # 设备1
        self.tracker2 = ClientStatsTracker(app_state)  # 设备2
        self.lock = threading.Lock()
        self.records = []  # 合并的记录
    
    def log(self, message: str, device: int = 0):
        """记录日志（带设备标识）"""
        device_tag = f"[Dev{device}]" if device > 0 else ""
        full_message = f"{device_tag} {message}"
        print(full_message)
        
        if self.app_state:
            with self.lock:
                timestamp = time.strftime('%H:%M:%S')
                self.app_state.logs.append(f"[{timestamp}] {full_message}")
                if len(self.app_state.logs) > 500:
                    self.app_state.logs = self.app_state.logs[-300:]
    
    def get_tracker(self, device: int) -> ClientStatsTracker:
        """获取指定设备的跟踪器"""
        return self.tracker1 if device == 1 else self.tracker2
    
    def merge_records(self):
        """合并两个设备的记录"""
        self.records = self.tracker1.records + self.tracker2.records
        self.records.sort(key=lambda x: x['client_id'])
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取合并后的统计数据"""
        self.merge_records()
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)
    
    def save_to_excel(self, filepath: str = "dual_device_stats.xlsx"):
        """保存合并的统计数据"""
        self.merge_records()
        if not self.records:
            self.log("没有统计数据可保存")
            return
        
        df = pd.DataFrame(self.records)
        
        # 添加设备标识列
        df['device'] = df['client_id'].apply(lambda x: 1 if x % 2 == 0 else 2)
        
        # 汇总行
        summary_row = {
            'client_id': 'TOTAL/AVG',
            'device': 'ALL',
            'training_time_s': df['training_time_s'].mean(),
            'upload_time_s': df['upload_time_s'].mean(),
            'upload_bytes': df['upload_bytes'].sum(),
            'upload_KB': df['upload_KB'].sum(),
            'upload_MB': df['upload_MB'].sum(),
        }
        
        df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_with_summary.to_excel(writer, sheet_name='Dual Device Stats', index=False)
        
        self.log(f"✓ 统计数据已保存到 {filepath}")


def flush_and_wait_for_ready(
    ser: serial.Serial, 
    dual_tracker=None, 
    device_id=None, 
    deadline=None,
    quiet_time: float = 5.0,      # 需要保持安静的时间（秒）
    max_wait_time: float = 600   # 最大等待时间（秒）
) -> bool:
    """
    清空缓冲区并等待设备稳定（不再有数据输出）
    
    Args:
        ser: 串口对象
        dual_tracker: 日志记录器
        device_id: 设备ID
        deadline: 全局超时时间点
        quiet_time: 设备需要保持安静的时间（无新数据）
        max_wait_time: 最大等待时间
        
    Returns:
        True: 设备已稳定
        False: 超时或失败
    """
    device_tag = f"[设备{device_id}]" if device_id else ""
    
    def log(msg):
        if dual_tracker:
            dual_tracker.log(f"{device_tag} {msg}")
    
    try:
        # 第一步：清空当前缓冲区
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        log("已清空缓冲区，等待设备安静...")
        
        start_time = time.time()
        last_data_time = time.time()  # 上次收到数据的时间
        total_discarded = 0
        
        while True:
            # 检查全局超时
            if deadline and time.time() > deadline:
                log(f"等待设备安静时全局超时")
                return False
            
            # 检查最大等待时间
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                log(f"等待设备安静超时 ({max_wait_time}s)，已丢弃 {total_discarded} 字节")
                return False
            
            # 检查是否有新数据
            if ser.in_waiting > 0:
                # 有数据，读取并丢弃
                data = ser.read(ser.in_waiting)
                total_discarded += len(data)
                last_data_time = time.time()  # 重置安静计时器
                
                # 可选：打印丢弃的数据（调试用）
                # log(f"丢弃 {len(data)} 字节: {data[:50]}...")
            else:
                # 没有新数据，检查是否已经安静足够长时间
                quiet_duration = time.time() - last_data_time
                if quiet_duration >= quiet_time:
                    log(f"设备已安静 {quiet_time}s，共丢弃 {total_discarded} 字节，可以重试")
                    return True
            
            # 短暂等待，避免CPU空转
            time.sleep(0.1)
            
    except serial.SerialException as e:
        log(f"串口错误: {e}")
        return False
    except Exception as e:
        log(f"等待设备安静时发生错误: {e}")
        return False



def run_single_device_worker(
    ser,
    device_id: int,
    pending_queue: list,  # ★★★ 改为共享队列
    queue_lock: threading.Lock,  # ★★★ 队列锁
    train_dataset: list,
    app_state: AppState,
    tracker: ClientStatsTracker,
    dual_tracker: DualDeviceTracker,
    stop_event: threading.Event,
    results: dict,
    results_lock: threading.Lock,
    scale: float = 100.0,
    max_retries: int = 2,
    global_timeout_seconds: float = GLOBAL_TIMEOUT_SECONDS
):
    """单个设备的工作线程（动态轮流模式）"""
    device_tag = f"[设备{device_id}]"
    dual_tracker.log(f"{device_tag} 启动，等待任务...", device_id)
    
    failed_clients = []
    processed_count = 0
    
    # ★★★ 从队列中动态获取客户端 ★★★
    while True:
        # 检查停止信号
        if stop_event.is_set():
            dual_tracker.log(f"{device_tag} 收到停止信号", device_id)
            break
        
        # 检查暂停
        while app_state.is_paused:
            if stop_event.is_set():
                break
            time.sleep(0.1)
        
        # ★★★ 从队列中获取下一个客户端 ★★★
        client_id = None
        with queue_lock:
            if pending_queue:
                client_id = pending_queue.pop(0)
                dual_tracker.log(f"{device_tag} 从队列获取客户端 {client_id} (剩余: {len(pending_queue)})", device_id)
            else:
                # 队列为空，退出
                dual_tracker.log(f"{device_tag} 队列为空，完成所有任务", device_id)
                break
        
        if client_id is None:
            break
        
        # ★★★ 更新设备当前客户端显示 ★★★
        if device_id == 1:
            app_state.device1_current_client = client_id
        elif device_id == 2:
            app_state.device2_current_client = client_id
        
        dual_tracker.log(f"\n{device_tag} {'=' * 40}", device_id)
        dual_tracker.log(f"{device_tag} 处理客户端 {client_id}", device_id)
        dual_tracker.log(f"{device_tag} {'=' * 40}", device_id)
        
        ds = train_dataset[client_id]
        data_int16 = prepare_client_data_int16(ds, scale=scale)

        # 重试逻辑（保持不变）
        success = False
        last_error = None
        
        for attempt in range(max_retries + 1):
            # 每次尝试使用新的截止时间
            deadline = make_deadline(global_timeout_seconds)
            if attempt > 0:
                dual_tracker.log(f"\n{device_tag} ══════════════════════════════", device_id)
                dual_tracker.log(f"{device_tag} 🔄 客户端 {client_id} 第 {attempt} 次重试", device_id)
                dual_tracker.log(f"{device_tag} ══════════════════════════════", device_id)
                
                if client_id < len(app_state.clients):
                    app_state.clients[client_id].retry_count = attempt
                    app_state.clients[client_id].status = f"{get_status_text('status_retrying')} ({attempt}/{max_retries})"
                    app_state.clients[client_id].progress = 5
                
                # 重试前重置，给予 30s 的独立超时时间
                try:
                    reset_deadline = make_deadline(30.0)
                    flush_and_wait_for_ready(ser, dual_tracker, device_id, reset_deadline)
                except TimeoutError as e:
                    dual_tracker.log(f"{device_tag} 重置超时（继续重试）: {e}", device_id)
                except Exception as e:
                    dual_tracker.log(f"{device_tag} 重置异常（继续重试）: {e}", device_id)
            
            try:
                tracker.start_client(client_id)
                
                W, XTX = send_client_data_for_device(
                    ser, client_id, data_int16, tracker, app_state, device_id, dual_tracker, deadline
                )
                
                with results_lock:
                    results['W'][client_id] = W
                    results['XTX'][client_id] = XTX
                
                if client_id < len(app_state.clients):
                    app_state.clients[client_id].error_msg = None
                
                if attempt > 0:
                    dual_tracker.log(f"{device_tag} ✓ 客户端 {client_id} 重试成功！", device_id)
                
                success = True
                processed_count += 1
                break
            
            except TimeoutError as e:
                last_error = str(e)
                dual_tracker.log(f"{device_tag} ✗ 超时: {e}", device_id)
                
                if attempt == max_retries:
                    tracker.mark_error(get_status_text('timeout_with_retries').format(max_retries))
                else:
                    if client_id < len(app_state.clients):
                        app_state.clients[client_id].status = get_status_text('status_waiting_retry')
                        app_state.clients[client_id].error_msg = None
                    dual_tracker.log(f"{device_tag} 超时冷却 5s...", device_id)
                    time.sleep(5.0)
            
            except Exception as e:
                last_error = str(e)
                dual_tracker.log(f"{device_tag} ✗ 错误: {e}", device_id)
                
                if attempt == max_retries:
                    tracker.mark_error(f"{e} (已重试{max_retries}次)")
                else:
                    if client_id < len(app_state.clients):
                        app_state.clients[client_id].status = get_status_text('status_waiting_retry')
                        app_state.clients[client_id].error_msg = None
                    dual_tracker.log(f"{device_tag} 异常冷却 5s...", device_id)
                    time.sleep(5.0)
        
        if not success:
            failed_clients.append(client_id)
            dual_tracker.log(f"{device_tag} ✗ 客户端 {client_id} 最终失败: {last_error}", device_id)
            # 最终失败后重置，给予 30s 的独立超时时间
            try:
                reset_deadline = make_deadline(30.0)
                flush_and_wait_for_ready(ser, dual_tracker, device_id, reset_deadline)
            except TimeoutError as e:
                dual_tracker.log(f"{device_tag} 重置超时（忽略继续）: {e}", device_id)
            except Exception as e:
                dual_tracker.log(f"{device_tag} 重置异常（忽略继续）: {e}", device_id)
        
        dual_tracker.log(f"{device_tag} 休息 10s 防过热", device_id)
        time.sleep(10.0)
    
    # ★★★ 设备完成所有任务后清除显示 ★★★
    if device_id == 1:
        app_state.device1_current_client = -1
        dual_tracker.log(f"{device_tag} 已清除当前客户端显示", device_id)
    elif device_id == 2:
        app_state.device2_current_client = -1
        dual_tracker.log(f"{device_tag} 已清除当前客户端显示", device_id)
    
    # 汇报结果
    dual_tracker.log(f"\n{device_tag} ════════════════════════════════", device_id)
    dual_tracker.log(f"{device_tag} 成功处理: {processed_count} 个客户端", device_id)
    if failed_clients:
        dual_tracker.log(f"{device_tag} 失败: {failed_clients}", device_id)
    dual_tracker.log(f"{device_tag} ════════════════════════════════", device_id)






def send_client_data_for_device(
    ser, 
    client_id: int, 
    data_int16, 
    tracker: ClientStatsTracker, 
    app_state: AppState,
    device_id: int,
    dual_tracker: DualDeviceTracker,
    deadline: Optional[float]
):
    """为双设备模式封装的发送函数（带 SLIP 转义）"""
    num_samples, num_features = data_int16.shape
    device_tag = f"[设备{device_id}]"

    # ★★★ 新增：更新设备当前客户端 ★★★
    if device_id == 1:
        app_state.device1_current_client = client_id
    elif device_id == 2:
        app_state.device2_current_client = client_id
    
    dual_tracker.log(f"{device_tag} 准备发送: {num_samples} x {num_features}", device_id)
    tracker.start_client(client_id)
    
    try:
        # 发送命令
        check_deadline(deadline, f"{device_tag} 发送CLIENT_BEGIN", tracker)
        ser.write(f"CLIENT_BEGIN {client_id}\n".encode())
        ser.flush()
        time.sleep(0.1)
        
        check_deadline(deadline, f"{device_tag} 发送SCALE", tracker)
        ser.write(f"SCALE {app_state.scale}\n".encode())
        ser.flush()
        time.sleep(0.1)
        
        check_deadline(deadline, f"{device_tag} 发送SAMPLES", tracker)
        ser.write(f"SAMPLES {num_samples}\n".encode())
        ser.flush()
        time.sleep(0.1)
        
        check_deadline(deadline, f"{device_tag} 发送FEATURES", tracker)
        ser.write(f"FEATURES {num_features}\n".encode())
        ser.flush()
        time.sleep(0.2)
        
        tracker.update_client_status(get_status_text('status_sending'), 10)
        
        # 等待准备
        check_deadline(deadline, f"{device_tag} 等待STM32准备", tracker)
        if not wait_for_response(ser, "ready to receive binary data", tracker, deadline):
            raise TimeoutError(f"{device_tag} STM32 未响应")
        
        dual_tracker.log(f"{device_tag} 开始发送...", device_id)
        tracker.update_client_status(get_status_text('status_sending'), 15)
        
        # ★★★ 使用 SLIP 转义 ★★★
        check_deadline(deadline, f"{device_tag} SLIP转义", tracker)
        raw = data_int16.astype('<i2', copy=False).tobytes(order='C')
        encoded_data, stats = slip_encode_with_stats(raw)
        
        dual_tracker.log(f"{device_tag} SLIP 转义: {stats['original_size']} → {stats['encoded_size']} 字节 "
                        f"(开销 {stats['overhead_percent']:.2f}%, 转义 {stats['total_escaped']} 个)", device_id)
        
        CHUNK_SIZE = 1024
        total_sent = 0
        
        while total_sent < len(encoded_data):
            check_deadline(deadline, f"{device_tag} 发送数据块", tracker)
            
            chunk = encoded_data[total_sent:total_sent + CHUNK_SIZE]
            ser.write(chunk)
            ser.flush()
            total_sent += len(chunk)
            
            if total_sent % 8192 == 0 or total_sent == len(encoded_data):
                progress = (total_sent / len(encoded_data)) * 100
                mapped_progress = 10 + (progress / 100) * 15
                tracker.update_client_status(get_status_text('status_sending'), mapped_progress)
            
            time.sleep(0.05)  # ★ 快速发送
        
        dual_tracker.log(f"{device_tag} 数据发送完成", device_id)
        tracker.update_client_status(get_status_text('status_processing'), 25)
        
        # 等待 READY
        check_deadline(deadline, f"{device_tag} 等待READY", tracker)
        if not wait_for_response(ser, "READY", tracker, deadline):
            raise TimeoutError(f"{device_tag} 未收到 READY")
        
        dual_tracker.log(f"{device_tag} ✓ 收到 READY", device_id)
        
        # 发送结束
        check_deadline(deadline, f"{device_tag} 发送CLIENT_END", tracker)
        time.sleep(0.2)
        ser.write(b"CLIENT_END\n")
        ser.flush()
        
        dual_tracker.log(f"{device_tag} 等待计算...", device_id)

        # 标记进入结果接收阶段，避免仍显示为训练中
        tracker.update_client_status(get_status_text('status_uploading'), 60)
        
        # 接收结果
        check_deadline(deadline, f"{device_tag} 接收W", tracker)
        W = recv_matrix_binary(ser, tracker, tag="W_SHAPE", deadline=deadline)
        
        check_deadline(deadline, f"{device_tag} 接收XTX", tracker)
        XTX = recv_xtx_streaming_binary(ser, tracker, deadline=deadline)
        
        tracker.finish_upload()
        tracker.finish_client()
        time.sleep(1)
        
        dual_tracker.log(f"{device_tag} ✓ 完成: W{W.shape}, XTX{XTX.shape}", device_id)
        
        return W, XTX
        
    except TimeoutError as e:
        dual_tracker.log(f"{device_tag} ✗ 超时: {e}", device_id)
        tracker.mark_error(str(e))
        raise
        
    except Exception as e:
        dual_tracker.log(f"{device_tag} ✗ 错误: {e}", device_id)
        tracker.mark_error(str(e))
        raise



def run_dual_device_parallel(
    train_dataset: list,
    app_state: AppState,
    dual_state: DualDeviceState,
    stop_event: threading.Event,
    save_path: str = "./output/dual_results.npz",
    stats_excel_path: str = "./output/dual_stats.xlsx",
    global_timeout_seconds: float = GLOBAL_TIMEOUT_SECONDS
):
    """
    ★★★ 双设备并行处理主函数 ★★★
    
    - 设备1 处理偶数客户端: 0, 2, 4, 6, ...
    - 设备2 处理奇数客户端: 1, 3, 5, 7, ...
    - 两个设备并行运行
    """
    num_clients = len(train_dataset)
    output_dir = os.path.dirname(save_path) or "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建双设备跟踪器
    dual_tracker = DualDeviceTracker(app_state)
    
    dual_tracker.log("=" * 60)
    dual_tracker.log("★★★ 双设备并行模式启动 ★★★")
    dual_tracker.log(f"设备1: {dual_state.device1_port} (偶数客户端)")
    dual_tracker.log(f"设备2: {dual_state.device2_port} (奇数客户端)")
    dual_tracker.log(f"总客户端数: {num_clients}")
    dual_tracker.log("=" * 60)
    
    # ★★★ 动态队列模式：初始化待处理队列 ★★★
    pending_queue = list(range(num_clients))  # [0, 1, 2, 3, ..., num_clients-1]
    queue_lock = threading.Lock()

    dual_tracker.log(f"待处理队列: {pending_queue}")
    dual_tracker.log("模式: 动态轮流分配（哪个设备先完成，哪个处理下一个）")

    
    # 打开两个串口
    ser1 = None
    ser2 = None
    
    try:
        dual_tracker.log(f"\n[设备1] 打开串口 {dual_state.device1_port}...", 1)
        ser1 = open_serial(dual_state.device1_port, dual_state.device1_baudrate)
        dual_state.device1_connected = True
        dual_tracker.log(f"[设备1] ✓ 串口已打开", 1)
        
        dual_tracker.log(f"\n[设备2] 打开串口 {dual_state.device2_port}...", 2)
        ser2 = open_serial(dual_state.device2_port, dual_state.device2_baudrate)
        dual_state.device2_connected = True
        dual_tracker.log(f"[设备2] ✓ 串口已打开", 2)
        
    except Exception as e:
        dual_tracker.log(f"[Error] 串口打开失败: {e}")
        if ser1:
            ser1.close()
        if ser2:
            ser2.close()
        app_state.is_running = False
        raise
    
    # 共享结果存储
    results = {
        'W': [None] * num_clients,
        'XTX': [None] * num_clients
    }
    results_lock = threading.Lock()
    
    # ★★★ 创建两个工作线程（共享队列）★★★
    thread1 = threading.Thread(
        target=run_single_device_worker,
        args=(ser1, 1, pending_queue, queue_lock, train_dataset, app_state,  # ← 传入共享队列
            dual_tracker.tracker1, dual_tracker, stop_event, 
            results, results_lock, dual_state.scale, 2, global_timeout_seconds),
        name="Device1-Worker"
    )

    thread2 = threading.Thread(
        target=run_single_device_worker,
        args=(ser2, 2, pending_queue, queue_lock, train_dataset, app_state,  # ← 传入共享队列
            dual_tracker.tracker2, dual_tracker, stop_event,
            results, results_lock, dual_state.scale, 2, global_timeout_seconds),
        name="Device2-Worker"
    )

    
    try:
        # 启动两个线程
        dual_tracker.log("\n启动并行处理...")
        start_time = time.time()
        
        thread1.start()
        thread2.start()
        
        # 等待两个线程完成
        thread1.join()
        thread2.join()
        
        elapsed = time.time() - start_time
        dual_tracker.log(f"\n{'=' * 60}")
        dual_tracker.log(f"★★★ 双设备并行处理完成 ★★★")
        dual_tracker.log(f"总耗时: {elapsed:.2f}s")
        dual_tracker.log(f"{'=' * 60}")
        
        # 整理结果（过滤 None）
        all_W = [w for w in results['W'] if w is not None]
        all_XTX = [x for x in results['XTX'] if x is not None]
        
        dual_tracker.log(f"成功完成: {len(all_W)}/{num_clients} 个客户端")
        
        # 保存结果
        if all_W:
            np.savez(save_path, W=all_W, XTX=all_XTX)
            dual_tracker.log(f"✓ 结果已保存: {save_path}")
        
        # 保存统计
        dual_tracker.save_to_excel(stats_excel_path)
        
        return all_W, all_XTX, results
        
    except KeyboardInterrupt:
        dual_tracker.log("\n用户中断 (Ctrl+C)")
        stop_event.set()
        thread1.join(timeout=2)
        thread2.join(timeout=2)
        dual_tracker.save_to_excel(stats_excel_path)
        
    except Exception as e:
        dual_tracker.log(f"\n错误: {e}")
        stop_event.set()
        dual_tracker.save_to_excel(stats_excel_path)
        raise
        
    finally:
        app_state.is_running = False
        if ser1:
            ser1.close()
            dual_tracker.log("[设备1] 串口已关闭", 1)
        if ser2:
            ser2.close()
            dual_tracker.log("[设备2] 串口已关闭", 2)


def test_dual_serial_connection(
    port1: str, baudrate1: int,
    port2: str, baudrate2: int
) -> dict:
    """测试双串口连接"""
    result = {
        'device1': {'success': False, 'message': ''},
        'device2': {'success': False, 'message': ''},
        'overall_success': False
    }
    
    # 测试设备1
    result['device1'] = test_serial_connection(port1, baudrate1)
    
    # 测试设备2
    result['device2'] = test_serial_connection(port2, baudrate2)
    
    # 检查是否是同一个串口
    if port1 == port2:
        result['device1']['success'] = False
        result['device2']['success'] = False
        result['device1']['message'] = "错误: 两个设备不能使用同一个串口"
        result['device2']['message'] = "错误: 两个设备不能使用同一个串口"
    
    result['overall_success'] = result['device1']['success'] and result['device2']['success']
    
    return result


# ============================
# ★★★ 双设备模拟测试 ★★★
# ============================

def run_dual_simulation(
    num_clients: int,
    app_state: AppState,
    stop_event: threading.Event
):
    """
    双设备模拟运行（用于测试 UI，无需真实串口）
    """
    import random
    
    dual_tracker = DualDeviceTracker(app_state)
    
    dual_tracker.log("=" * 50)
    dual_tracker.log("[Simulation] ★★★ 双设备模拟测试 ★★★")
    dual_tracker.log(f"[Simulation] 模拟 {num_clients} 个客户端")
    dual_tracker.log("=" * 50)
    
    # ★★★ 动态队列模式 ★★★
    pending_queue = list(range(num_clients))
    queue_lock = threading.Lock()

    dual_tracker.log(f"待处理队列: {pending_queue}")
    dual_tracker.log("模式: 动态轮流分配")

    
    results_lock = threading.Lock()
    
    def simulate_device(device_id, tracker):  # ★★★ 去掉 client_ids 参数
        device_tag = f"[设备{device_id}]"
        processed = 0
        
        # ★★★ 从队列动态获取客户端 ★★★
        while True:
            if stop_event.is_set():
                dual_tracker.log(f"{device_tag} 收到停止信号", device_id)
                break
            
            while app_state.is_paused:
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            
            # ★★★ 从共享队列获取下一个客户端 ★★★
            client_id = None
            with queue_lock:
                if pending_queue:
                    client_id = pending_queue.pop(0)
                    dual_tracker.log(f"{device_tag} 获取客户端 {client_id} (剩余: {len(pending_queue)})", device_id)
                else:
                    # 队列为空，退出
                    dual_tracker.log(f"{device_tag} 队列为空，完成所有任务", device_id)
                    break
            
            if client_id is None:
                break
            
            # ★★★ 更新设备当前客户端 ★★★
            if device_id == 1:
                app_state.device1_current_client = client_id
            elif device_id == 2:
                app_state.device2_current_client = client_id
            
            dual_tracker.log(f"{device_tag} 处理客户端 {client_id}", device_id)
            tracker.start_client(client_id)
            
            # 模拟发送
            tracker.update_client_status(get_status_text('status_sending'), 10)
            time.sleep(0.3)
            
            # 模拟训练
            tracker.training_start_time = time.time()
            tracker.update_client_status(get_status_text('status_training'), 30)
            train_time = random.uniform(1.0, 2.5)
            
            for i in range(5):
                if stop_event.is_set():
                    break
                progress = 30 + i * 8
                tracker.update_client_status(get_status_text('status_training'), progress)
                time.sleep(train_time / 5)
            
            tracker.training_end_time = time.time()
            if client_id < len(app_state.clients):
                app_state.clients[client_id].training_time = train_time
            
            # 模拟上传
            tracker.update_client_status(get_status_text('status_uploading'), 70)
            upload_time = random.uniform(0.5, 1.5)
            upload_bytes = random.randint(50000, 150000)
            
            for i in range(5):
                if stop_event.is_set():
                    break
                progress = 70 + i * 6
                tracker.update_client_status(get_status_text('status_uploading'), progress)
                tracker.add_binary_data(upload_bytes // 5)
                time.sleep(upload_time / 5)
            
            tracker.upload_end_time = time.time()
            if client_id < len(app_state.clients):
                app_state.clients[client_id].upload_time = upload_time
            
            tracker.finish_client()
            dual_tracker.log(f"{device_tag} ✓ 客户端 {client_id} 完成", device_id)
            
            processed += 1
            time.sleep(0.3)
        
        # ★★★ 清除设备当前客户端显示 ★★★
        if device_id == 1:
            app_state.device1_current_client = -1
        elif device_id == 2:
            app_state.device2_current_client = -1
        
        dual_tracker.log(f"{device_tag} 成功处理 {processed} 个客户端", device_id)


    # 创建两个模拟线程
    thread1 = threading.Thread(
        target=simulate_device,
        args=(1, dual_tracker.tracker1)  # ← 不再传递 even_clients
    )
    thread2 = threading.Thread(
        target=simulate_device,
        args=(2, dual_tracker.tracker2)  # ← 不再传递 odd_clients
    )
    
    start_time = time.time()
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    elapsed = time.time() - start_time
    
    dual_tracker.log("\n" + "=" * 50)
    dual_tracker.log(f"[Simulation] ✓ 双设备模拟完成")
    dual_tracker.log(f"[Simulation] 总耗时: {elapsed:.2f}s")
    dual_tracker.log("=" * 50)
    
    dual_tracker.save_to_excel("./output/dual_simulation_stats.xlsx")
    
    app_state.is_running = False
    
    return dual_tracker


