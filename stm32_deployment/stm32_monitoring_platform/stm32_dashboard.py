import streamlit as st
import pandas as pd
import time
import threading
import os
from datetime import datetime

# 尝试导入 plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 导入串口模块
# 导入串口模块
from stm32_serial import (
    AppState, ClientStats, ClientStatsTracker,
    run_all_clients_with_autosave,
    run_simulation,
    load_client_data_from_excel,
    test_serial_connection,
    get_available_ports,
    save_checkpoint,
    # ★★★ 新增导入 ★★★
    DualDeviceState,
    DualDeviceTracker,
    run_dual_device_parallel,
    run_dual_simulation,
    test_dual_serial_connection
)


# ============================
# 多语言支持
# ============================
LANGUAGES = {
    'zh': {
        # 页面标题
        'page_title': 'STM32 联邦学习监控',
        'main_title': '📡 STM32 联邦学习监控平台',
        'main_subtitle': '实时监控嵌入式设备的训练进度和通信状态',
        
        # 侧边栏
        'control_panel': '⚙️ 控制面板',
        'running': '● 运行中',
        'stopped': '○ 已停止',
        'paused': '⏸ 暂停中',
        
        # 数据加载
        'data_loading': '📂 数据加载',
        'data_dir': '📁 数据目录',
        'data_dir_help': '包含 client_*_data.xlsx 文件的目录',
        'load_data': '📂 加载数据',
        'loading': '加载中...',
        'loaded_clients': '✓ 已加载 {} 个客户端',
        'load_failed': '加载失败: {}',
        
        # 运行模式
        'run_mode': '🚀 运行模式',
        'device_mode': '🔧 设备模式',
        'single_device': '单设备',
        'dual_device': '双设备并行',
        'device_mode_help': '单设备：使用一个STM32；双设备：两个STM32并行处理',
        'run_type': '🎯 运行类型',
        'simulation': '🧪 模拟测试',
        'real_serial': '📡 真实串口',
        'run_type_help': '模拟测试：无需硬件；真实串口：连接STM32',
        'sim_clients': '👥 模拟客户端数量',
        'sim_clients_help': '模拟测试的客户端数量',
        'clients_loaded': '📊 已加载 {} 个客户端数据',
        'load_data_first': '⚠️ 请先在「数据加载」中加载数据',
        'device1_count': '📍 设备1: {}个| 设备2: {}个',
        
        # 串口配置
        'serial_config': '🔌 串口配置',
        'serial_config_dual': '🔌 串口配置 (双设备)',
        'device1': '📟 设备1 ',
        'device2': '📟 设备2',
        'serial_port': '🔌 串口',
        'serial1': '🔌 串口1',
        'serial2': '🔌 串口2',
        'baudrate': '📶 波特率',
        'scale': '📐 缩放系数',
        'scale_shared': '📐 缩放系数 (共享)',
        'scale_help': '数据量化缩放因子 (SCALE)',
        'scale_shared_help': '两个设备共享的数据量化缩放因子',
        'refresh_ports': '🔄 刷新串口',
        'test_connection': '🔌 测试连接',
        'test_serial': '🔌 测试串口连接',
        'testing': '测试中...',
        'only_one_port': '⚠️ 只检测到一个串口',
        'same_port_error': '❌ 两个设备不能使用同一个串口！',
        'device1_connected': '✓ 设备1 已连接',
        'device1_not_connected': '○ 设备1 未连接',
        'device2_connected': '✓ 设备2 已连接',
        'device2_not_connected': '○ 设备2 未连接',
        'serial_port_help': '选择STM32连接的串口',
        'serial_port_manual_help': '手动输入串口号，如 COM4',
        
        # 输出设置
        'output_settings': '📁 输出设置',
        'output_dir': '💾 输出目录',
        'output_dir_help': '结果和检查点的保存位置',
        'auto_save_tip': '💡 中断时自动保存到此目录',
        'dual_save_tip': '💡 双设备模式结果保存为 dual_results.npz',
        
        # 控制按钮
        'run_control': '🎮 运行控制',
        'start': '▶ 开始',
        'stop': '⏹ 停止',
        'pause': '⏸ 暂停',
        'resume': '▶ 继续',
        'manual_save': '💾 手动保存',
        'start_simulation': '开始模拟测试',
        'start_dual_run': '开始双设备并行运行',
        'start_run': '开始运行',
        'load_data_tooltip': '请先加载数据',
        'connect_device1_tooltip': '请先连接设备1',
        'connect_device2_tooltip': '请先连接设备2',
        'same_port_tooltip': '两个设备串口不能相同',
        'test_serial_tooltip': '请先测试串口连接',
        'stop_and_save': '停止运行并自动保存',
        'resume_tooltip': '继续运行',
        'pause_tooltip': '暂停运行',
        'save_now': '立即保存当前结果',
        'stopped_saved': '✓ 已停止并保存',
        
        # 进度摘要
        'progress_summary': '📊 进度摘要',
        'device1_even': '**设备1 **',
        'device2_odd': '**设备2 **',
        'device1_even_': '**设备1 已完成**',
        'device2_odd_': '**设备2 已完成**',
        'completed': '✅ 完成',
        'in_progress': '🔄 进行中',
        'error': '❌ 错误',
        'waiting_start': '等待开始...',
        
        # 刷新设置
        'refresh_settings': '🔄 界面刷新',
        'auto_refresh': '自动刷新',
        'auto_refresh_help': '运行时自动刷新界面',
        'interval': '间隔',
        'interval_help': '刷新间隔（秒）',
        
        # 当前配置
        'current_config': '📋 当前配置',
        'config_item': '配置项',
        'config_value': '值',
        'mode': '模式',
        'device1_port': '设备1串口',
        'device1_baudrate': '设备1波特率',
        'device2_port': '设备2串口',
        'device2_baudrate': '设备2波特率',
        'output_directory': '输出目录',
        'run_type_label': '运行类型',
        'client_count': '客户端数',
        'simulation_label': '模拟',
        'real_serial_label': '真实串口',
        
        # 主页面指标
        'status': '状态',
        'status_running': '🟢 运行中',
        'status_stopped': '🔴 停止',
        'status_paused': '🟡 暂停',
        'progress': '进度',
        'avg_training': '平均训练',
        'avg_upload': '平均上传',
        'total_data': '总数据',
        
        # 客户端进度
        'client_progress': '📊 客户端进度',
        'click_start': '👆 点击侧边栏「开始」按钮启动处理',
        'total_progress': '总进度: {}/{} 客户端完成',
        'retry_success': '🔄 重试{}次后成功',
        'retrying': '⏳ 重试中...',
        'error_label': '错误: {}',
        
        # 统计图表
        'statistics': '📈 统计图表',
        'training_time': '⏱ 训练时间',
        'upload_time': '📤 上传时间',
        'data_distribution': '📦 数据量分布',
        'comprehensive_stats': '📊 综合统计',
        'training_time_chart': '各客户端训练时间',
        'upload_time_chart': '各客户端上传时间',
        'device_data_compare': '设备数据量对比',
        'data_ratio': '数据量占比',
        'no_stats_data': '暂无统计数据，等待客户端完成...',
        'no_stats': '暂无统计数据',
        'device1_label': '设备1',
        'device2_label': '设备2',
        'completed_count': '完成: {} 个',
        'avg_training_time': '平均训练: {:.2f}s',
        'total_data_kb': '总数据: {:.1f} KB',
        'total_clients': '总客户端数',
        'avg_training_time_label': '平均训练时间',
        'avg_upload_time_label': '平均上传时间',
        'total_data_label': '总数据量',
        'fastest_training': '最快训练',
        'slowest_training': '最慢训练',
        'client_id': '客户端ID',
        'time_s': '时间(s)',
        
        # 详细数据
        'detailed_data': '📋 详细数据',
        'client_status': '客户端状态',
        'id': 'ID',
        'status_col': '状态',
        'progress_col': '进度',
        'training_time_col': '训练时间',
        'upload_time_col': '上传时间',
        'upload_bytes_col': '上传字节',
        'upload_kb_col': '上传KB',
        'upload_mb_col': '上传MB',
        'total_avg_label': '汇总/平均',
        'data_size': '数据量',
        'error_col': '错误',
        'stats_records': '统计记录',
        'download_csv': '📥 下载CSV',
        'download_excel': '📥 下载Excel',
        
        # 日志
        'logs': '📜 运行日志',
        'clear_logs': '🗑 清空日志',
        'download_logs': '📥 下载日志',
        'no_logs': '暂无日志...',
        
        # 已保存文件
        'saved_files': '💾 已保存文件',
        'output_dir_not_exist': '输出目录 {} 不存在',
        'no_saved_files': '暂无保存的文件',
        'filename': '文件名',
        'file_size': '大小',
        'modified_time': '修改时间',
        'download_files': '下载文件',
        'select_file': '选择文件',
        'download_file': '📥 下载 {}',
        
        # 标签页
        'tab_progress': '📊 进度监控',
        'tab_charts': '📈 统计图表',
        'tab_data': '📋 详细数据',
        'tab_logs': '📜 日志',
        'tab_files': '💾 已保存文件',
        
        # 活跃客户端
        'processing': '🔥 正在处理',
        'more_processing': '还有 {} 个正在处理...',
        'waiting_client': '⏳ 等待客户端开始...',
        'idle': '💤 空闲中',
        
        # 系统消息
        'stop_signal': '[System] 收到停止信号...',
        'stopped_auto_saved': '[System] ✓ 已停止（数据已自动保存）',
        'no_data_to_save': '没有可保存的数据',
        
        # 语言切换
        'language': '🌐 语言',
        'chinese': '中文',
        'english': 'English',
        
        # 动态状态文本
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

        # 错误/超时消息
        'wait_tag_timeout': '等待 {} 超时',
        'stm32_no_response': 'STM32 未响应',
        'stm32_no_ready': 'STM32 未响应 READY',
        'stm32_not_sending_ready': 'STM32 未发送 READY',
        'timeout_with_retries': '超时 (已重试{}次)',
        'global_timeout_stage': '全局超时 ({})',
        
        # 串口测试消息
        'serial_test_success': '连接成功: {}',
        'serial_test_opened': '串口 {} 已打开（未收到响应）',
        'serial_test_error': '串口错误: {}',
        'serial_test_unknown_error': '未知错误: {}',
    },
    'en': {
        # Page title
        'page_title': 'STM32 Federated Learning Monitor',
        'main_title': '📡 STM32 Federated Learning Monitor',
        'main_subtitle': 'Real-time monitoring of embedded device training progress and communication status',
        
        # Sidebar
        'control_panel': '⚙️ Control Panel',
        'running': '● Running',
        'stopped': '○ Stopped',
        'paused': '⏸ Paused',
        
        # Data loading
        'data_loading': '📂 Data Loading',
        'data_dir': '📁 Data Directory',
        'data_dir_help': 'Directory containing client_*_data.xlsx files',
        'load_data': '📂 Load Data',
        'loading': 'Loading...',
        'loaded_clients': '✓ Loaded {} clients',
        'load_failed': 'Load failed: {}',
        
        # Run mode
        'run_mode': '🚀 Run Mode',
        'device_mode': '🔧 Device Mode',
        'single_device': 'Single Device',
        'dual_device': 'Dual Device Parallel',
        'device_mode_help': 'Single: Use one STM32; Dual: Two STM32s in parallel',
        'run_type': '🎯 Run Type',
        'simulation': '🧪 Simulation',
        'real_serial': '📡 Real Serial',
        'run_type_help': 'Simulation: No hardware needed; Real Serial: Connect STM32',
        'sim_clients': '👥 Simulated Clients',
        'sim_clients_help': 'Number of simulated clients',
        'clients_loaded': '📊 Loaded {} client data',
        'load_data_first': '⚠️ Please load data first in "Data Loading"',
        'device1_count': '📍 Device1: {} | Device2: {}',
        
        # Serial config
        'serial_config': '🔌 Serial Config',
        'serial_config_dual': '🔌 Serial Config (Dual)',
        'device1': '📟 Device1',
        'device2': '📟 Device2',
        'serial_port': '🔌 Port',
        'serial1': '🔌 Port1',
        'serial2': '🔌 Port2',
        'baudrate': '📶 Baudrate',
        'scale': '📐 Scale',
        'scale_shared': '📐 Scale (Shared)',
        'scale_help': 'Data quantization scale factor (SCALE)',
        'scale_shared_help': 'Shared scale factor for both devices',
        'refresh_ports': '🔄 Refresh Ports',
        'test_connection': '🔌 Test Connection',
        'test_serial': '🔌 Test Serial',
        'testing': 'Testing...',
        'only_one_port': '⚠️ Only one port detected',
        'same_port_error': '❌ Two devices cannot use the same port!',
        'device1_connected': '✓ Device1 Connected',
        'device1_not_connected': '○ Device1 Not Connected',
        'device2_connected': '✓ Device2 Connected',
        'device2_not_connected': '○ Device2 Not Connected',
        'serial_port_help': 'Select the serial port connected to STM32',
        'serial_port_manual_help': 'Enter port manually, e.g., COM4',
        
        # Output settings
        'output_settings': '📁 Output Settings',
        'output_dir': '💾 Output Directory',
        'output_dir_help': 'Save location for results and checkpoints',
        'auto_save_tip': '💡 Auto-save to this directory on interrupt',
        'dual_save_tip': '💡 Dual device results saved as dual_results.npz',
        
        # Control buttons
        'run_control': '🎮 Run Control',
        'start': '▶ Start',
        'stop': '⏹ Stop',
        'pause': '⏸ Pause',
        'resume': '▶ Resume',
        'manual_save': '💾 Manual Save',
        'start_simulation': 'Start simulation',
        'start_dual_run': 'Start dual device parallel run',
        'start_run': 'Start running',
        'load_data_tooltip': 'Please load data first',
        'connect_device1_tooltip': 'Please connect device1 first',
        'connect_device2_tooltip': 'Please connect device2 first',
        'same_port_tooltip': 'Two devices cannot use the same port',
        'test_serial_tooltip': 'Please test serial connection first',
        'stop_and_save': 'Stop and auto-save',
        'resume_tooltip': 'Resume running',
        'pause_tooltip': 'Pause running',
        'save_now': 'Save current results now',
        'stopped_saved': '✓ Stopped and saved',
        
        # Progress summary
        'progress_summary': '📊 Progress Summary',
        'device1_even': '**Device1 **',
        'device2_odd': '**Device2 **',
        'device1_even_': '**Device1 completed**',
        'device2_odd_': '**Device2 completed**',
        'completed': '✅ Done',
        'in_progress': '🔄 Running',
        'error': '❌ Error',
        'waiting_start': 'Waiting to start...',
        
        # Refresh settings
        'refresh_settings': '🔄 UI Refresh',
        'auto_refresh': 'Auto Refresh',
        'auto_refresh_help': 'Auto refresh UI while running',
        'interval': 'Interval',
        'interval_help': 'Refresh interval (seconds)',
        
        # Current config
        'current_config': '📋 Current Config',
        'config_item': 'Config',
        'config_value': 'Value',
        'mode': 'Mode',
        'device1_port': 'Device1 Port',
        'device1_baudrate': 'Device1 Baudrate',
        'device2_port': 'Device2 Port',
        'device2_baudrate': 'Device2 Baudrate',
        'output_directory': 'Output Directory',
        'run_type_label': 'Run Type',
        'client_count': 'Client Count',
        'simulation_label': 'Simulation',
        'real_serial_label': 'Real Serial',
        
        # Main page metrics
        'status': 'Status',
        'status_running': '🟢 Running',
        'status_stopped': '🔴 Stopped',
        'status_paused': '🟡 Paused',
        'progress': 'Progress',
        'avg_training': 'Avg Training',
        'avg_upload': 'Avg Upload',
        'total_data': 'Total Data',
        
        # Client progress
        'client_progress': '📊 Client Progress',
        'click_start': '👆 Click "Start" button in sidebar to begin',
        'total_progress': 'Total: {}/{} clients completed',
        'retry_success': '🔄 Succeeded after {} retries',
        'retrying': '⏳ Retrying...',
        'error_label': 'Error: {}',
        
        # Statistics charts
        'statistics': '📈 Statistics',
        'training_time': '⏱ Training Time',
        'upload_time': '📤 Upload Time',
        'data_distribution': '📦 Data Distribution',
        'comprehensive_stats': '📊 Summary Stats',
        'training_time_chart': 'Training Time per Client',
        'upload_time_chart': 'Upload Time per Client',
        'device_data_compare': 'Device Data Comparison',
        'data_ratio': 'Data Ratio',
        'no_stats_data': 'No statistics yet, waiting for clients to complete...',
        'no_stats': 'No statistics available',
        'device1_label': 'Device1',
        'device2_label': 'Device2',
        'completed_count': 'Completed: {}',
        'avg_training_time': 'Avg Training: {:.2f}s',
        'total_data_kb': 'Total Data: {:.1f} KB',
        'total_clients': 'Total Clients',
        'avg_training_time_label': 'Avg Training Time',
        'avg_upload_time_label': 'Avg Upload Time',
        'total_data_label': 'Total Data',
        'fastest_training': 'Fastest Training',
        'slowest_training': 'Slowest Training',
        'client_id': 'Client ID',
        'time_s': 'Time(s)',
        
        # Detailed data
        'detailed_data': '📋 Detailed Data',
        'client_status': 'Client Status',
        'id': 'ID',
        'status_col': 'Status',
        'progress_col': 'Progress',
        'training_time_col': 'Training Time',
        'upload_time_col': 'Upload Time',
        'upload_bytes_col': 'Upload Bytes',
        'upload_kb_col': 'Upload KB',
        'upload_mb_col': 'Upload MB',
        'total_avg_label': 'TOTAL/AVG',
        'data_size': 'Data Size',
        'error_col': 'Error',
        'stats_records': 'Statistics Records',
        'download_csv': '📥 Download CSV',
        'download_excel': '📥 Download Excel',
        
        # Logs
        'logs': '📜 Run Logs',
        'clear_logs': '🗑 Clear Logs',
        'download_logs': '📥 Download Logs',
        'no_logs': 'No logs yet...',
        
        # Saved files
        'saved_files': '💾 Saved Files',
        'output_dir_not_exist': 'Output directory {} does not exist',
        'no_saved_files': 'No saved files yet',
        'filename': 'Filename',
        'file_size': 'Size',
        'modified_time': 'Modified',
        'download_files': 'Download Files',
        'select_file': 'Select file',
        'download_file': '📥 Download {}',
        
        # Tabs
        'tab_progress': '📊 Progress',
        'tab_charts': '📈 Charts',
        'tab_data': '📋 Data',
        'tab_logs': '📜 Logs',
        'tab_files': '💾 Files',
        
        # Active clients
        'processing': '🔥 Processing',
        'more_processing': '{} more processing...',
        'waiting_client': '⏳ Waiting for clients to start...',
        'idle': '💤 Idle',
        
        # System messages
        'stop_signal': '[System] Stop signal received...',
        'stopped_auto_saved': '[System] ✓ Stopped (data auto-saved)',
        'no_data_to_save': 'No data to save',
        
        # Language switch
        'language': '🌐 Language',
        'chinese': '中文',
        'english': 'English',
        
        # Dynamic status texts
        'status_waiting': 'Waiting',
        'status_preparing': 'Preparing',
        'status_training': 'Training',
        'status_uploading': 'Uploading',
        'status_sending': 'Sending Data',
        'status_processing': 'Processing',
        'status_completed': 'Done ✓',
        'status_error': 'Error ✗',
        'status_retrying': 'Retrying',
        'status_waiting_retry': 'Waiting to Retry',

        # Error/timeout messages
        'wait_tag_timeout': 'Timeout waiting for {}',
        'stm32_no_response': 'STM32 did not respond',
        'stm32_no_ready': 'STM32 did not send READY',
        'stm32_not_sending_ready': 'STM32 did not send READY',
        'timeout_with_retries': 'Timeout (retried {} times)',
        'global_timeout_stage': 'Global timeout ({})',
        
        # Serial test messages
        'serial_test_success': 'Connected: {}',
        'serial_test_opened': 'Port {} opened (no response)',
        'serial_test_error': 'Serial error: {}',
        'serial_test_unknown_error': 'Unknown error: {}',
    }
}

def get_text(key):
    """获取当前语言的文本"""
    lang = st.session_state.get('language', 'en')
    return LANGUAGES.get(lang, LANGUAGES['en']).get(key, key)

def t(key):
    """get_text的简写"""
    return get_text(key)

def is_completed_status(status: str) -> bool:
    """判断是否为完成状态（支持中英文）"""
    return status in [get_text('status_completed'), "完成 ✓", "Done ✓"]

def is_error_status(status: str) -> bool:
    """判断是否为错误状态（支持中英文）"""
    return "错误" in status or "Error" in status or get_text('status_error') in status

def is_waiting_status(status: str) -> bool:
    """判断是否为等待状态（支持中英文）"""
    return status in [get_text('status_waiting'), "等待中", "Waiting"]


def is_client_done(client) -> bool:
    """判断客户端是否已完成（容错：进度已满也视为完成）"""
    if client is None:
        return False
    if getattr(client, 'progress', 0) >= 99.9:
        return True
    return is_completed_status(client.status)


def is_client_active(client) -> bool:
    """正在处理的客户端判定（排除已完成/错误/等待）"""
    if client is None:
        return False
    if is_client_done(client) or is_error_status(client.status) or is_waiting_status(client.status):
        return False
    return getattr(client, 'progress', 0) < 99.9


def normalize_display_status(client) -> str:
    """规范化展示状态：进度满但未标记完成时强制显示完成"""
    display_status = to_display_status(client.status)
    if getattr(client, 'progress', 0) >= 99.9 and not is_completed_status(display_status):
        return get_text('status_completed')
    return display_status


# 状态标准化：确保语言切换后显示使用当前语言
def get_status_variants():
    return {
        'status_waiting': [get_text('status_waiting'), "等待中", "Waiting"],
        'status_preparing': [get_text('status_preparing'), "准备中", "Preparing"],
        'status_training': [get_text('status_training'), "训练中", "Training"],
        'status_uploading': [get_text('status_uploading'), "上传中", "Uploading"],
        'status_sending': [get_text('status_sending'), "发送数据中", "Sending Data"],
        'status_processing': [get_text('status_processing'), "等待处理", "Processing"],
        'status_completed': [get_text('status_completed'), "完成 ✓", "Done ✓"],
        'status_error': [get_text('status_error'), "错误 ✗", "Error ✗"],
        'status_retrying': [get_text('status_retrying'), "重试中", "Retrying"],
        'status_waiting_retry': [get_text('status_waiting_retry'), "准备重试", "Waiting to Retry"],
    }


def to_display_status(status: str) -> str:
    """将任意语言的状态转换为当前语言显示"""
    # ★★★ 修复：更健壮地处理重试状态（支持带括号的格式如 "重试中 (1/2)" 或 "Retrying (1/2)"）★★★
    # 定义所有可能的重试状态前缀（硬编码，避免依赖动态获取可能产生的问题）
    retry_prefixes = ["重试中", "Retrying"]
    for prefix in retry_prefixes:
        if status.startswith(prefix):
            suffix = status[len(prefix):]
            return f"{get_text('status_retrying')}{suffix}"
    
    # ★★★ 同样处理"准备重试"/"Waiting to Retry"等状态 ★★★
    waiting_retry_prefixes = ["准备重试", "Preparing retry", "Waiting to Retry"]
    for prefix in waiting_retry_prefixes:
        if status.startswith(prefix):
            suffix = status[len(prefix):]
            return f"{get_text('status_waiting_retry')}{suffix}"

    # 处理其他标准状态
    for key, variants in get_status_variants().items():
        if status in variants:
            return get_text(key)
    return status


def is_retrying_status(status: str) -> bool:
    """判断是否为重试相关状态"""
    variants = get_status_variants()['status_retrying']
    return status in variants or "重试" in status or "Retry" in status

# ============================
# 页面配置
# ============================
st.set_page_config(
    page_title="STM32 Federated Learning Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ★★★ 优化的CSS样式 ★★★
# ============================
st.markdown("""
<style>
    /* ★★★ 只隐藏右上角运行状态指示器（烟花/New图标）★★★ */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ★★★ 客户端列表容器自适应高度 ★★★ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        max-height: calc(100vh - 420px) !important;
        min-height: 300px !important;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        max-height: calc(100vh - 420px) !important;
        min-height: 300px !important;
    }

    /* 侧边栏按钮统一大小 */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        height: 36px;
        padding: 0 10px;
        font-size: 13px;
        margin: 2px 0;
    }
    
    /* 主按钮样式 */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #00cc66;
        color: white;
    }
    
    /* 侧边栏紧凑间距 */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    
    section[data-testid="stSidebar"] hr {
        margin: 0.5rem 0;
    }
    
    section[data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h4 {
        font-size: 0.9rem;
        margin: 0.3rem 0;
    }
    
    /* 输入框紧凑 */
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        padding: 6px 10px;
        font-size: 13px;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        min-height: 32px;
    }
    
    /* Expander紧凑 */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        font-size: 14px;
        padding: 8px 0;
    }
    
    /* 状态徽章 */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 500;
    }
    .status-running { background: #d4edda; color: #155724; }
    .status-stopped { background: #f8d7da; color: #721c24; }
    .status-paused { background: #fff3cd; color: #856404; }
    
    /* 进度条颜色 */
    .stProgress > div > div > div > div {
        background-color: #00cc66;
    }
    
    /* 侧边栏语言选择器紧凑样式 */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"][aria-label="🌐"] > div > div {
        min-height: 28px;
        font-size: 12px;
    }
    
    /* Metric紧凑 */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================
# Session State 初始化
# ============================
def init_session_state():
    defaults = {
        'app_state': AppState(),
        'tracker': None,
        'stop_event': threading.Event(),
        'worker_thread': None,
        'train_dataset': None,
        'data_loaded': False,
        'output_dir': "./output",
        'all_W': [],
        'all_XTXs': [],
        # ★★★ 新增双设备状态 ★★★
        'dual_state': DualDeviceState(),
        'dual_mode': False,  # 是否启用双设备模式
        # ★★★ 语言设置 ★★★
        'language': 'en',  # Default to English
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()



# ============================
# 工作线程
# ============================
def worker_function(app_state, tracker, stop_event, mode, train_dataset, num_clients, output_dir, dual_state=None):
    """后台工作线程"""
    try:
        if mode == "simulation":
            run_simulation(num_clients, app_state, tracker, stop_event)
        elif mode == "dual_simulation":
            # ★★★ 双设备模拟 ★★★
            run_dual_simulation(num_clients, app_state, stop_event)
        elif mode == "dual_real":
            # ★★★ 双设备真实模式 ★★★
            if dual_state is None:
                raise ValueError("双设备模式需要提供 dual_state 参数")
            run_dual_device_parallel(
                train_dataset=train_dataset,
                app_state=app_state,
                dual_state=dual_state,
                stop_event=stop_event,
                save_path=os.path.join(output_dir, "dual_results.npz"),
                stats_excel_path=os.path.join(output_dir, "dual_stats.xlsx")
            )
        else:
            # 单设备模式（原有逻辑）
            all_W, all_XTXs, _ = run_all_clients_with_autosave(
                train_dataset=train_dataset,
                app_state=app_state,
                tracker=tracker,
                stop_event=stop_event,
                save_path=os.path.join(output_dir, "results.npz"),
                stats_excel_path=os.path.join(output_dir, "client_stats.xlsx"),
                checkpoint_interval=5
            )
            st.session_state.all_W = all_W
            st.session_state.all_XTXs = all_XTXs
            
    except Exception as e:
        if tracker:
            tracker.log(f"[Error] 线程异常: {e}")
        if st.session_state.all_W:
            save_checkpoint(
                st.session_state.all_W, 
                st.session_state.all_XTXs, 
                tracker, 
                output_dir, 
                f"异常: {e}"
            )
    finally:
        app_state.is_running = False



def start_worker(mode, num_clients, train_dataset=None):
    """启动工作线程"""
    app_state = st.session_state.app_state
    output_dir = st.session_state.output_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 重置状态
    app_state.is_running = True
    app_state.is_paused = False
    app_state.total_clients = num_clients
    app_state.clients = [ClientStats(client_id=i) for i in range(num_clients)]
    app_state.logs = []
    
    # 重置结果
    st.session_state.all_W = []
    st.session_state.all_XTXs = []
    
    # 创建跟踪器（单设备模式才需要）
    if mode in ["simulation", "real"]:
        st.session_state.tracker = ClientStatsTracker(app_state=app_state)
    else:
        st.session_state.tracker = None  # 双设备模式使用 DualDeviceTracker
    
    # 重置停止事件
    st.session_state.stop_event.clear()
     # ★★★ 修复：在主线程中获取 dual_state，然后传递给工作线程 ★★★
    dual_state = None
    if mode in ["dual_real", "dual_simulation"]:
        dual_state = st.session_state.dual_state
    
    # 启动线程
    st.session_state.worker_thread = threading.Thread(
        target=worker_function,
        args=(app_state, st.session_state.tracker, st.session_state.stop_event,
              mode, train_dataset, num_clients, output_dir, dual_state),
        daemon=True
    )
    st.session_state.worker_thread.start()



def stop_worker():
    """停止工作线程（会触发自动保存）"""
    tracker = st.session_state.tracker
    app_state = st.session_state.app_state
    
    if tracker:
        tracker.log(t('stop_signal'))
    
    # 设置停止事件 - run_all_clients_with_autosave 会自动保存
    st.session_state.stop_event.set()
    
    # 等待一小段时间让线程处理
    time.sleep(0.5)
    
    app_state.is_running = False
    
    if tracker:
        tracker.log(t('stopped_auto_saved'))


def manual_save():
    """手动保存当前结果"""
    tracker = st.session_state.tracker
    output_dir = st.session_state.output_dir
    
    if not st.session_state.all_W and not (tracker and tracker.records):
        return False, t('no_data_to_save')
    
    success, msg = save_checkpoint(
        st.session_state.all_W,
        st.session_state.all_XTXs,
        tracker,
        output_dir,
        "Manual Save" if st.session_state.get('language', 'en') == 'en' else "手动保存"
    )
    return success, msg


# ============================
# ★★★ 优化后的侧边栏 ★★★
# ============================
# ============================
# ★★★ 优化后的侧边栏（带标签）★★★
# ============================
def render_sidebar():
    app_state = st.session_state.app_state
    dual_state = st.session_state.dual_state
    
    with st.sidebar:
        # ===== 语言切换 =====
        col_title, col_lang = st.columns([3, 2])
        with col_title:
            st.title(t('control_panel'))
        with col_lang:
            current_lang = st.session_state.get('language', 'en')
            lang_options = ['zh', 'en']
            lang_labels = {'zh': '中文', 'en': 'EN'}
            new_lang = st.selectbox(
                "🌐",
                options=lang_options,
                index=lang_options.index(current_lang),
                format_func=lambda x: lang_labels[x],
                key='sidebar_lang_selector',
                label_visibility='collapsed'
            )
            if new_lang != current_lang:
                st.session_state.language = new_lang
                st.rerun()
        
        # ===== 运行状态指示 =====
        if app_state.is_running:
            if app_state.is_paused:
                st.markdown('<span class="status-badge status-paused">' + t('paused') + '</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-running">' + t('running') + '</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-stopped">' + t('stopped') + '</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # ===== 1. 数据加载 =====
        with st.expander(t('data_loading'), expanded=not st.session_state.data_loaded):
            data_dir = st.text_input(
                t('data_dir'),
                value="./output/client_data",
                help=t('data_dir_help')
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(t('load_data'), use_container_width=True, 
                            disabled=app_state.is_running):
                    try:
                        with st.spinner(t('loading')):
                            st.session_state.train_dataset = load_client_data_from_excel(data_dir)
                            st.session_state.data_loaded = True
                        st.success(t('loaded_clients').format(len(st.session_state.train_dataset)))
                        st.rerun()
                    except Exception as e:
                        st.error(t('load_failed').format(e))
            with col2:
                if st.session_state.data_loaded:
                    st.success(f"✓ {len(st.session_state.train_dataset)}")
        
        # ===== 2. 运行模式选择 =====
        with st.expander(t('run_mode'), expanded=True):
            # ★★★ 新增：设备模式选择 ★★★
            device_mode = st.radio(
                t('device_mode'),
                [t('single_device'), t('dual_device')],
                horizontal=True,
                help=t('device_mode_help')
            )
            st.session_state.dual_mode = (device_mode == t('dual_device'))
            
            st.divider()
            
            run_mode = st.radio(
                t('run_type'),
                [t('simulation'), t('real_serial')],
                horizontal=True,
                help=t('run_type_help')
            )
            is_simulation = t('simulation') in run_mode or "模拟" in run_mode or "Simulation" in run_mode
            
            if is_simulation:
                num_clients = st.slider(
                    t('sim_clients'),
                    min_value=2 if st.session_state.dual_mode else 1,
                    max_value=50,
                    value=10 if st.session_state.dual_mode else 5,
                    help=t('sim_clients_help')
                )
            else:
                if st.session_state.data_loaded:
                    num_clients = len(st.session_state.train_dataset)
                    st.info(t('clients_loaded').format(num_clients))
                else:
                    num_clients = 0
                    st.warning(t('load_data_first'))
            
            # # 双设备模式下显示任务分配
            # if st.session_state.dual_mode and num_clients > 0:
            #     even_count = len([i for i in range(num_clients) if i % 2 == 0])
            #     odd_count = len([i for i in range(num_clients) if i % 2 == 1])
            #     st.caption(t('device1_count').format(even_count, odd_count))
        
        # ===== 3. 串口配置 =====
        if st.session_state.dual_mode:
            # ★★★ 双设备串口配置 ★★★
            with st.expander(t('serial_config_dual'), expanded=True):
                available_ports = get_available_ports()
                
                # 设备1配置
                st.markdown(f"##### {t('device1')}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    if available_ports:
                        port1_idx = 0
                        if dual_state.device1_port in available_ports:
                            port1_idx = available_ports.index(dual_state.device1_port)
                        dual_state.device1_port = st.selectbox(
                            t('serial1'),
                            available_ports,
                            index=port1_idx,
                            key="port1",
                            help=t('serial_port_help')
                        )
                    else:
                        dual_state.device1_port = st.text_input(
                            t('serial1'),
                            value=dual_state.device1_port,
                            key="port1_input"
                        )
                with col2:
                    baud_options = [115200, 230400, 460800, 921600]
                    baud1_idx = baud_options.index(dual_state.device1_baudrate) if dual_state.device1_baudrate in baud_options else 2
                    dual_state.device1_baudrate = st.selectbox(
                        t('baudrate'),
                        baud_options,
                        index=baud1_idx,
                        key="baud1"
                    )
                
                st.divider()
                
                # 设备2配置
                st.markdown(f"##### {t('device2')}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    if available_ports:
                        # 默认选择不同于设备1的串口
                        remaining_ports = [p for p in available_ports if p != dual_state.device1_port]
                        if remaining_ports:
                            port2_idx = 0
                            if dual_state.device2_port in remaining_ports:
                                port2_idx = remaining_ports.index(dual_state.device2_port)
                            dual_state.device2_port = st.selectbox(
                                t('serial2'),
                                remaining_ports,
                                index=port2_idx,
                                key="port2",
                                help=t('serial_port_help')
                            )
                        else:
                            st.warning(t('only_one_port'))
                            dual_state.device2_port = st.text_input(
                                t('serial2'),
                                value=dual_state.device2_port,
                                key="port2_input"
                            )
                    else:
                        dual_state.device2_port = st.text_input(
                            t('serial2'),
                            value=dual_state.device2_port,
                            key="port2_input2"
                        )
                with col2:
                    baud2_idx = baud_options.index(dual_state.device2_baudrate) if dual_state.device2_baudrate in baud_options else 2
                    dual_state.device2_baudrate = st.selectbox(
                        t('baudrate'),
                        baud_options,
                        index=baud2_idx,
                        key="baud2"
                    )
                
                st.divider()
                
                # 共享缩放系数
                dual_state.scale = st.number_input(
                    t('scale_shared'),
                    value=dual_state.scale,
                    min_value=1.0,
                    max_value=10000.0,
                    step=10.0,
                    help=t('scale_shared_help')
                )
                app_state.scale = dual_state.scale  # 同步到 app_state
                
                # 检查串口是否相同
                if dual_state.device1_port == dual_state.device2_port:
                    st.error(t('same_port_error'))
                
                # 刷新和测试按钮
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t('refresh_ports'), use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button(t('test_connection'), use_container_width=True, disabled=app_state.is_running):
                        with st.spinner(t('testing')):
                            result = test_dual_serial_connection(
                                dual_state.device1_port, dual_state.device1_baudrate,
                                dual_state.device2_port, dual_state.device2_baudrate
                            )
                            st.session_state['dual_last_test'] = result

                            if result['device1']['success']:
                                dual_state.device1_connected = True
                            else:
                                dual_state.device1_connected = False

                            if result['device2']['success']:
                                dual_state.device2_connected = True
                            else:
                                dual_state.device2_connected = False

                # 测试结果（单独占满一行，避免挤在右侧）
                test_result = st.session_state.get('dual_last_test') if 'dual_last_test' in st.session_state else None
                if test_result:
                    result_cols = st.columns(2)

                    with result_cols[0]:
                        if test_result['device1']['success']:
                            st.success(f"✓ {t('device1_label')}: {test_result['device1']['message']}")
                        else:
                            st.error(f"✗ {t('device1_label')}: {test_result['device1']['message']}")

                    with result_cols[1]:
                        if test_result['device2']['success']:
                            st.success(f"✓ {t('device2_label')}: {test_result['device2']['message']}")
                        else:
                            st.error(f"✗ {t('device2_label')}: {test_result['device2']['message']}")
                
                # 连接状态指示
                col1, col2 = st.columns(2)
                with col1:
                    if dual_state.device1_connected:
                        st.success(t('device1_connected'))
                    else:
                        st.caption(t('device1_not_connected'))
                with col2:
                    if dual_state.device2_connected:
                        st.success(t('device2_connected'))
                    else:
                        st.caption(t('device2_not_connected'))
        
        else:
            # ★★★ 单设备串口配置（原有逻辑）★★★
            with st.expander(t('serial_config'), expanded=True):
                available_ports = get_available_ports()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if available_ports:
                        port_idx = 0
                        if app_state.serial_port in available_ports:
                            port_idx = available_ports.index(app_state.serial_port)
                        app_state.serial_port = st.selectbox(
                            t('serial_port'),
                            available_ports,
                            index=port_idx,
                            help=t('serial_port_help')
                        )
                    else:
                        app_state.serial_port = st.text_input(
                            t('serial_port'),
                            value=app_state.serial_port,
                            help=t('serial_port_manual_help')
                        )
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("🔄", help=t('refresh_ports'), use_container_width=True):
                        st.rerun()
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    baud_options = [115200, 230400, 460800, 921600]
                    baud_idx = baud_options.index(app_state.baudrate) if app_state.baudrate in baud_options else 2
                    app_state.baudrate = st.selectbox(
                        t('baudrate'),
                        baud_options,
                        index=baud_idx,
                        help=t('baudrate')
                    )
                with col2:
                    app_state.scale = st.number_input(
                        t('scale'),
                        value=app_state.scale,
                        min_value=1.0,
                        max_value=10000.0,
                        step=10.0,
                        help=t('scale_help')
                    )
                
                if st.button(t('test_serial'), use_container_width=True, disabled=app_state.is_running):
                    with st.spinner(t('testing')):
                        result = test_serial_connection(app_state.serial_port, app_state.baudrate)
                        if result['success']:
                            app_state.serial_connected = True
                            st.success(f"✓ {result['message']}")
                        else:
                            app_state.serial_connected = False
                            st.error(f"✗ {result['message']}")
        
        # ===== 4. 输出设置 =====
        with st.expander(t('output_settings'), expanded=False):
            st.session_state.output_dir = st.text_input(
                t('output_dir'),
                value=st.session_state.output_dir,
                help=t('output_dir_help')
            )
            st.caption(t('auto_save_tip'))
            if st.session_state.dual_mode:
                st.caption(t('dual_save_tip'))
        
        st.divider()
        
        # ===== 5. 控制按钮 =====
        st.markdown(f"#### {t('run_control')}")
        
        # 计算开始按钮是否可用
        if is_simulation:
            start_disabled = app_state.is_running
            start_tooltip = t('start_simulation')
        else:
            if st.session_state.dual_mode:
                # 双设备模式
                start_disabled = (app_state.is_running or
                                 not st.session_state.data_loaded or
                                 not dual_state.device1_connected or
                                 not dual_state.device2_connected or
                                 dual_state.device1_port == dual_state.device2_port)
                if not st.session_state.data_loaded:
                    start_tooltip = t('load_data_tooltip')
                elif not dual_state.device1_connected:
                    start_tooltip = t('connect_device1_tooltip')
                elif not dual_state.device2_connected:
                    start_tooltip = t('connect_device2_tooltip')
                elif dual_state.device1_port == dual_state.device2_port:
                    start_tooltip = t('same_port_tooltip')
                else:
                    start_tooltip = t('start_dual_run')
            else:
                # 单设备模式
                start_disabled = (app_state.is_running or
                                 not st.session_state.data_loaded or
                                 not app_state.serial_connected)
                if not st.session_state.data_loaded:
                    start_tooltip = t('load_data_tooltip')
                elif not app_state.serial_connected:
                    start_tooltip = t('test_serial_tooltip')
                else:
                    start_tooltip = t('start_run')
        
        # 开始/停止按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t('start'), type="primary", use_container_width=True,
                        disabled=start_disabled, help=start_tooltip):
                # 确定运行模式
                if is_simulation:
                    if st.session_state.dual_mode:
                        mode = "dual_simulation"
                    else:
                        mode = "simulation"
                else:
                    if st.session_state.dual_mode:
                        mode = "dual_real"
                    else:
                        mode = "real"
                
                start_worker(mode, num_clients, st.session_state.train_dataset if not is_simulation else None)
                st.rerun()
        
        with col2:
            if st.button(t('stop'), type="secondary", use_container_width=True,
                        disabled=not app_state.is_running,
                        help=t('stop_and_save')):
                stop_worker()
                st.toast(t('stopped_saved'), icon="💾")
                st.rerun()
        
        # 暂停/保存按钮
        col1, col2 = st.columns(2)
        with col1:
            pause_text = t('resume') if app_state.is_paused else t('pause')
            pause_help = t('resume_tooltip') if app_state.is_paused else t('pause_tooltip')
            if st.button(pause_text, use_container_width=True,
                        disabled=not app_state.is_running, help=pause_help):
                app_state.is_paused = not app_state.is_paused
                st.rerun()
        
        with col2:
            if st.button(t('manual_save'), use_container_width=True,
                        help=t('save_now')):
                success, msg = manual_save()
                if success:
                    st.toast(f"✓ {msg}", icon="💾")
                else:
                    st.toast(f"✗ {msg}", icon="⚠️")
        
        st.divider()
        
        # ===== 6. 实时进度摘要 =====
        st.markdown(f"#### {t('progress_summary')}")
        
        if app_state.clients:
            completed = sum(1 for c in app_state.clients if is_client_done(c))
            errors = sum(1 for c in app_state.clients if is_error_status(c.status))
            running = sum(1 for c in app_state.clients if is_client_active(c))
            
            # # 双设备模式下显示分组统计
            # if st.session_state.dual_mode:
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         st.markdown(t('device1_even'))
            #         even_completed = sum(1 for i, c in enumerate(app_state.clients) if i % 2 == 0 and is_client_done(c))
            #         even_total = len([i for i in range(len(app_state.clients)) if i % 2 == 0])
            #         st.caption(f"✅ {even_completed}/{even_total}")
            #     with col2:
            #         st.markdown(t('device2_odd'))
            #         odd_completed = sum(1 for i, c in enumerate(app_state.clients) if i % 2 == 1 and is_client_done(c))
            #         odd_total = len([i for i in range(len(app_state.clients)) if i % 2 == 1])
            #         st.caption(f"✅ {odd_completed}/{odd_total}")
            # if st.session_state.dual_mode:
            #     # 从日志或状态中统计每个设备实际处理的数量
            #     device1_processed = sum(1 for c in app_state.clients if is_client_done(c) and c.client_id in device1_actual_list)  # 需要实际记录
            #     device2_processed = sum(1 for c in app_state.clients if is_client_done(c) and c.client_id in device2_actual_list)
                
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         st.markdown(t('device1_even_'))
            #         st.caption(f"✅ {device1_processed} 个")
            #     with col2:
            #         st.markdown(t('device2_odd_'))
            #         st.caption(f"✅ {device2_processed} 个")

            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('completed'), completed)
            with col2:
                st.metric(t('in_progress'), running)
            with col3:
                st.metric(t('error'), errors)
            
            # 总进度条
            progress = completed / app_state.total_clients if app_state.total_clients > 0 else 0
            st.progress(progress, text=f"{completed}/{app_state.total_clients}")
        else:
            st.caption(t('waiting_start'))
        
        st.divider()
        
        # ===== 7. 刷新设置 =====
        st.markdown(f"#### {t('refresh_settings')}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            auto_refresh = st.checkbox(
                t('auto_refresh'),
                value=True,
                help=t('auto_refresh_help')
            )
        with col2:
            refresh_interval = st.select_slider(
                t('interval'),
                options=[0.5, 1.0, 2.0, 3.0, 5.0],
                value=1.0,
                help=t('interval_help')
            )
        
        # ===== 8. 当前配置摘要 =====
        with st.expander(t('current_config'), expanded=False):
            if st.session_state.dual_mode:
                st.markdown(f"""
                | {t('config_item')} | {t('config_value')} |
                |--------|-----|
                | {t('mode')} | `{t('dual_device')}` |
                | {t('device1_port')} | `{dual_state.device1_port}` |
                | {t('device1_baudrate')} | `{dual_state.device1_baudrate}` |
                | {t('device2_port')} | `{dual_state.device2_port}` |
                | {t('device2_baudrate')} | `{dual_state.device2_baudrate}` |
                | {t('scale')} | `{dual_state.scale}` |
                | {t('output_directory')} | `{st.session_state.output_dir}` |
                | {t('run_type_label')} | `{t('simulation_label') if is_simulation else t('real_serial_label')}` |
                | {t('client_count')} | `{num_clients}` |
                """)
            else:
                st.markdown(f"""
                | {t('config_item')} | {t('config_value')} |
                |--------|-----|
                | {t('mode')} | `{t('single_device')}` |
                | {t('serial_port')} | `{app_state.serial_port}` |
                | {t('baudrate')} | `{app_state.baudrate}` |
                | {t('scale')} | `{app_state.scale}` |
                | {t('output_directory')} | `{st.session_state.output_dir}` |
                | {t('run_type_label')} | `{t('simulation_label') if is_simulation else t('real_serial_label')}` |
                | {t('client_count')} | `{num_clients}` |
                """)
        
        return auto_refresh, refresh_interval, num_clients, is_simulation



# ============================
# 页面组件
# ============================
def render_header():
    """渲染头部（标题 + 活跃客户端）"""
    app_state = st.session_state.app_state
    
    # ★★★ 注入呼吸动画 CSS ★★★
    st.markdown("""
    <style>
    @keyframes breathing {
        0% { box-shadow: 0 0 5px rgba(0, 255, 136, 0.4); border-color: rgba(0, 255, 136, 0.6); }
        50% { box-shadow: 0 0 15px rgba(0, 255, 136, 0.9); border-color: rgba(0, 255, 136, 1); }
        100% { box-shadow: 0 0 5px rgba(0, 255, 136, 0.4); border-color: rgba(0, 255, 136, 0.6); }
    }
    
    .active-mini-card {
        border: 2px solid #00ff88;
        border-radius: 8px;
        padding: 8px 12px;
        animation: breathing 2s ease-in-out infinite;
        background: rgba(0, 255, 136, 0.08);
        text-align: center;
    }
    
    .active-mini-card .client-id {
        font-weight: bold;
        font-size: 0.9em;
        color: #fff;
    }
    
    .active-mini-card .client-status {
        font-size: 0.75em;
        color: #00ff88;
        margin: 4px 0;
    }
    
    .active-mini-card .progress-bar {
        background: #333;
        border-radius: 4px;
        height: 4px;
        overflow: hidden;
        margin-top: 6px;
    }
    
    .active-mini-card .progress-fill {
        background: linear-gradient(90deg, #00ff88, #00cc6a);
        height: 100%;
        transition: width 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ★★★ 标题和活跃客户端并排 ★★★
    col_title, col_active = st.columns([3, 2])
    
    with col_title:
        st.title(t('main_title'))
        st.caption(t('main_subtitle'))
    
    with col_active:
        # ★★★ 双设备模式：显示设备状态 ★★★
        if st.session_state.get('dual_mode', False):
            device1_active = app_state.device1_current_client if hasattr(app_state, 'device1_current_client') else -1
            device2_active = app_state.device2_current_client if hasattr(app_state, 'device2_current_client') else -1
            
            st.markdown(f"##### {t('processing')}")
            
            col_d1, col_d2 = st.columns(2)
            
            # 设备1状态
            with col_d1:
                if device1_active >= 0:
                    client = app_state.clients[device1_active] if device1_active < len(app_state.clients) else None
                    # ★★★ 修复：检查客户端是否已完成，已完成则显示休息中 ★★★
                    if client and not is_client_done(client):
                        display_status = normalize_display_status(client)
                        icon = "⚡" if display_status == get_text('status_training') else "📤" if display_status == get_text('status_uploading') else "📡" if display_status == get_text('status_sending') else "🔄"
                        
                        data_info = ""
                        # 只在上传中和已完成状态显示数据量
                        if display_status in [get_text('status_uploading'), get_text('status_completed')] and client.upload_bytes > 0:
                            if client.upload_bytes > 1024 * 1024:
                                data_info = f" | 📦{client.upload_bytes/1024/1024:.1f}MB"
                            else:
                                data_info = f" | 📦{client.upload_bytes/1024:.0f}KB"
                        
                        st.markdown(f"""
                        <div class="active-mini-card">
                            <div class="client-id">{icon} Client {client.client_id} (D1)</div>
                            <div class="client-status">{display_status}{data_info}</div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:{client.progress}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # 客户端已完成或不存在，显示休息中
                        st.info(f"💤 {t('device1_label')} {t('idle')}")
                else:
                     # 设备1空闲
                     st.info(f"💤 {t('device1_label')} {t('idle')}")  # ← 改这里 
            
            # 设备2状态
            with col_d2:
                if device2_active >= 0:
                    client = app_state.clients[device2_active] if device2_active < len(app_state.clients) else None
                    # ★★★ 修复：检查客户端是否已完成，已完成则显示休息中 ★★★
                    if client and not is_client_done(client):
                        display_status = normalize_display_status(client)
                        icon = "⚡" if display_status == get_text('status_training') else "📤" if display_status == get_text('status_uploading') else "📡" if display_status == get_text('status_sending') else "🔄"
                        
                        data_info = ""
                        # 只在上传中和已完成状态显示数据量
                        if display_status in [get_text('status_uploading'), get_text('status_completed')] and client.upload_bytes > 0:
                            if client.upload_bytes > 1024 * 1024:
                                data_info = f" | 📦{client.upload_bytes/1024/1024:.1f}MB"
                            else:
                                data_info = f" | 📦{client.upload_bytes/1024:.0f}KB"
                        
                        st.markdown(f"""
                        <div class="active-mini-card">
                            <div class="client-id">{icon} Client {client.client_id} (D2)</div>
                            <div class="client-status">{display_status}{data_info}</div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:{client.progress}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # 客户端已完成或不存在，显示休息中
                        st.info(f"💤 {t('device2_label')} {t('idle')}")
                else:
                    # 设备2空闲
                    st.info(f"💤 {t('device2_label')} {t('idle')}")
        
        else:
            # ★★★ 单设备模式：原有逻辑 ★★★
            active_clients = [c for c in app_state.clients if is_client_active(c)]
            
            if active_clients:
                st.markdown(f"##### {t('processing')}")
                
                # 最多显示2个活跃客户端
                display_clients = active_clients[:2]
                cols = st.columns(len(display_clients))
                
                for idx, client in enumerate(display_clients):
                    with cols[idx]:
                        display_status = normalize_display_status(client)
                        # 状态图标
                        if display_status == get_text('status_training'):
                            icon = "⚡"
                        elif display_status == get_text('status_uploading'):
                            icon = "📤"
                        elif display_status == get_text('status_sending'):
                            icon = "📡"
                        else:
                            icon = "🔄"
                        
                        # 设备标识
                        device_tag = ""
                        if st.session_state.get('dual_mode', False):
                            device_num = 1 if client.client_id % 2 == 0 else 2
                            device_tag = f" (D{device_num})"
                        
                        # 数据量显示
                        data_info = ""
                        if display_status in [get_text('status_uploading'), get_text('status_completed')]:
                            if client.upload_bytes > 0:
                                if client.upload_bytes > 1024 * 1024:
                                    data_info = f" | 📦{client.upload_bytes/1024/1024:.1f}MB"
                                else:
                                    data_info = f" | 📦{client.upload_bytes/1024:.0f}KB"
                        
                        st.markdown(f"""
                        <div class="active-mini-card">
                            <div class="client-id">{icon} Client {client.client_id}{device_tag}</div>
                            <div class="client-status">{display_status}{data_info}</div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:{client.progress}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 如果还有更多活跃客户端
                if len(active_clients) > 2:
                    st.caption(t('more_processing').format(len(active_clients) - 2))
            
            elif app_state.is_running:
                st.info(t('waiting_client'))
            
            else:
                st.caption(t('idle'))



def render_metrics():
    """渲染顶部指标"""
    app_state = st.session_state.app_state
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if app_state.is_running:
            status = t('status_running')
        else:
            status = t('status_stopped')
        if app_state.is_paused:
            status = t('status_paused')
        st.metric(t('status'), status)
    
    with col2:
        completed = sum(1 for c in app_state.clients if is_client_done(c))
        st.metric(t('progress'), f"{completed}/{app_state.total_clients}")
    
    with col3:
        # 只统计已完成的客户端
        times = [c.training_time for c in app_state.clients if c.training_time and c.training_time > 0]
        avg = sum(times) / len(times) if times else 0
        st.metric(t('avg_training'), f"{avg:.2f}s")
    
    with col4:
        # 只统计已完成的客户端
        times = [c.upload_time for c in app_state.clients if c.upload_time and c.upload_time > 0]
        avg = sum(times) / len(times) if times else 0
        st.metric(t('avg_upload'), f"{avg:.2f}s")
    
    with col5:
        # ★★★ 修复：实时统计所有客户端的数据量（包括正在上传的）★★★
        total = sum(c.upload_bytes for c in app_state.clients if c.upload_bytes)
        if total > 1024 * 1024:
            st.metric(t('total_data'), f"{total/1024/1024:.2f} MB")
        else:
            st.metric(t('total_data'), f"{total/1024:.1f} KB")



def render_progress():
    """渲染进度区域（干净版本，无重复）"""
    app_state = st.session_state.app_state
    
    st.subheader(t('client_progress'))
    
    if not app_state.clients:
        st.info(t('click_start'))
        return
    
    # ===== 总进度条 =====
    completed = sum(1 for c in app_state.clients if is_client_done(c))
    total_progress = completed / app_state.total_clients if app_state.total_clients > 0 else 0
    st.progress(total_progress, text=t('total_progress').format(completed, app_state.total_clients))
    
    st.divider()
    
    # ===== 可滚动的客户端列表 =====
    # ★★★ 使用较大的基础高度，CSS会覆盖为自适应高度 ★★★
    with st.container(height=600):
        cols_per_row = 4
        
        for i in range(0, len(app_state.clients), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(app_state.clients):
                    break
                
                client = app_state.clients[idx]
                display_status = normalize_display_status(client)
                
                with col:
                    # ★★★ 改进的状态图标逻辑 ★★★
                    if is_completed_status(display_status):
                        icon = "✅"
                    elif is_error_status(display_status) and not is_retrying_status(display_status):
                        icon = "❌"
                    elif is_retrying_status(display_status):
                        icon = "🔄"  # 重试中图标
                    elif display_status == get_text('status_training'):
                        icon = "⚡"
                    elif display_status == get_text('status_uploading'):
                        icon = "📤"
                    elif display_status == get_text('status_sending'):
                        icon = "📨"
                    elif display_status == get_text('status_waiting_retry') or "超时" in display_status or "Timeout" in display_status:
                        icon = "⏳"  # 等待重试
                    else:
                        icon = "⏳"
                    
                    st.markdown(f"**{icon} Client {client.client_id}**")
                    st.caption(display_status)
                    st.progress(client.progress / 100)
                    
                    # 统计信息
                    info_parts = []
                    if client.training_time and client.training_time > 0:
                        info_parts.append(f"🕐 {client.training_time:.1f}s")
                    if client.upload_time and client.upload_time > 0:
                        info_parts.append(f"📤 {client.upload_time:.1f}s")
                    # 仅在上传阶段或完成后显示数据量，避免训练中出现数据量错位
                    if display_status in [get_text('status_uploading'), get_text('status_completed')]:
                        if client.upload_bytes > 0:
                            if client.upload_bytes > 1024 * 1024:
                                info_parts.append(f"📦 {client.upload_bytes/1024/1024:.2f}MB")
                            else:
                                info_parts.append(f"📦 {client.upload_bytes/1024:.1f}KB")
                    
                    if info_parts:
                        st.caption(" | ".join(info_parts))
                    # ★★★ 在这里添加重试成功提示 ★★★
                    # 检查是否有 retry_count 属性
                    if hasattr(client, 'retry_count') and client.retry_count > 0:
                        if is_client_done(client):
                            st.success(t('retry_success').format(client.retry_count))
                    
                    # ★★★ 改进的错误显示逻辑 ★★★
                    if client.error_msg:
                        # 只有在非完成状态时才显示错误
                        if is_completed_status(display_status):
                            # 如果完成了但有 error_msg，说明是重试后成功的（理论上不应该发生）
                            pass  # 不显示，因为已经成功了
                        elif is_retrying_status(display_status):
                            # 正在重试，显示警告而不是错误
                            st.warning(t('retrying'))
                        else:
                            # 真正的错误
                            st.error(t('error_label').format(client.error_msg))






def render_charts():
    """渲染统计图表（支持单设备和双设备模式）"""
    app_state = st.session_state.app_state
    tracker = st.session_state.tracker
    dual_state = st.session_state.dual_state
    
    st.subheader(t('statistics'))
    
    # ★★★ 修复：双设备模式从 app_state.clients 获取数据 ★★★
    if st.session_state.dual_mode or tracker is None:
        # 双设备模式：从 app_state.clients 构建 DataFrame
        completed_clients = [c for c in app_state.clients if is_client_done(c)]
        
        if not completed_clients:
            st.info(t('no_stats_data'))
            return
        
        # 构建 DataFrame
        data = []
        for c in completed_clients:
            data.append({
                'client_id': c.client_id,
                'training_time_s': c.training_time if c.training_time else 0,
                'upload_time_s': c.upload_time if c.upload_time else 0,
                'upload_KB': c.upload_bytes / 1024 if c.upload_bytes else 0,
                'device': t('device1_label') if c.client_id % 2 == 0 else t('device2_label')  # 双设备模式下区分设备
            })
        
        df = pd.DataFrame(data)
        
    else:
        # 单设备模式：使用 tracker
        if not tracker.records:
            st.info(t('no_stats_data'))
            return
        df = tracker.get_dataframe()
    
    if df.empty:
        st.info(t('no_stats'))
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {t('training_time')}")
        if 'training_time_s' in df.columns:
            if PLOTLY_AVAILABLE:
                # ★★★ 双设备模式下按设备着色 ★★★
                if st.session_state.dual_mode and 'device' in df.columns:
                    fig = px.bar(df, x='client_id', y='training_time_s',
                                title=t('training_time_chart'),
                                labels={'client_id': t('client_id'), 'training_time_s': t('time_s')},
                                color='device',
                                color_discrete_map={t('device1_label'): '#2ecc71', t('device2_label'): '#3498db'})
                else:
                    fig = px.bar(df, x='client_id', y='training_time_s',
                                title=t('training_time_chart'),
                                labels={'client_id': t('client_id'), 'training_time_s': t('time_s')},
                                color='training_time_s',
                                color_continuous_scale='Greens')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index('client_id')['training_time_s'])
    
    with col2:
        st.markdown(f"#### {t('upload_time')}")
        if 'upload_time_s' in df.columns:
            if PLOTLY_AVAILABLE:
                if st.session_state.dual_mode and 'device' in df.columns:
                    fig = px.bar(df, x='client_id', y='upload_time_s',
                                title=t('upload_time_chart'),
                                labels={'client_id': t('client_id'), 'upload_time_s': t('time_s')},
                                color='device',
                                color_discrete_map={t('device1_label'): '#2ecc71', t('device2_label'): '#3498db'})
                else:
                    fig = px.bar(df, x='client_id', y='upload_time_s',
                                title=t('upload_time_chart'),
                                labels={'client_id': t('client_id'), 'upload_time_s': t('time_s')},
                                color='upload_time_s',
                                color_continuous_scale='Blues')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index('client_id')['upload_time_s'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {t('data_distribution')}")
        if 'upload_KB' in df.columns:
            if PLOTLY_AVAILABLE:
                if st.session_state.dual_mode and 'device' in df.columns:
                    # 双设备模式：按设备分组的饼图
                    device_data = df.groupby('device')['upload_KB'].sum().reset_index()
                    fig = px.pie(device_data, values='upload_KB', names='device',
                                title=t('device_data_compare'),
                                color='device',
                                color_discrete_map={t('device1_label'): '#2ecc71', t('device2_label'): '#3498db'})
                else:
                    fig = px.pie(df, values='upload_KB', names='client_id',
                                title=t('data_ratio'))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index('client_id')['upload_KB'])
    
    with col2:
        st.markdown(f"#### {t('comprehensive_stats')}")
        
        # ★★★ 双设备模式下显示分设备统计 ★★★
        if st.session_state.dual_mode and 'device' in df.columns:
            dev1_df = df[df['device'] == t('device1_label')]
            dev2_df = df[df['device'] == t('device2_label')]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(t('device1_even'))
                st.caption(t('completed_count').format(len(dev1_df)))
                if len(dev1_df) > 0:
                    st.caption(t('avg_training_time').format(dev1_df['training_time_s'].mean()))
                    st.caption(t('total_data_kb').format(dev1_df['upload_KB'].sum()))
            with col_b:
                st.markdown(t('device2_odd'))
                st.caption(t('completed_count').format(len(dev2_df)))
                if len(dev2_df) > 0:
                    st.caption(t('avg_training_time').format(dev2_df['training_time_s'].mean()))
                    st.caption(t('total_data_kb').format(dev2_df['upload_KB'].sum()))
            
            st.divider()
        
        summary = {
            t('total_clients'): len(df),
            t('avg_training_time_label'): f"{df['training_time_s'].mean():.2f}s" if 'training_time_s' in df.columns else "N/A",
            t('avg_upload_time_label'): f"{df['upload_time_s'].mean():.2f}s" if 'upload_time_s' in df.columns else "N/A",
            t('total_data_label'): f"{df['upload_KB'].sum():.2f} KB" if 'upload_KB' in df.columns else "N/A",
            t('fastest_training'): f"{df['training_time_s'].min():.2f}s" if 'training_time_s' in df.columns else "N/A",
            t('slowest_training'): f"{df['training_time_s'].max():.2f}s" if 'training_time_s' in df.columns else "N/A",
        }
        for key, value in summary.items():
            st.metric(key, value)


def render_table():
    """渲染详细数据表"""
    app_state = st.session_state.app_state
    tracker = st.session_state.tracker
    
    st.subheader(t('detailed_data'))
    
    # 客户端状态表
    if app_state.clients:
        st.markdown(f"#### {t('client_status')}")
        
        data = []
        for c in app_state.clients:
            display_status = normalize_display_status(c)
            data.append({
                t('id'): c.client_id,
                t('status_col'): display_status,
                t('progress_col'): f"{c.progress:.0f}%",
                t('training_time_col'): f"{c.training_time:.2f}s" if c.training_time else "-",
                t('upload_time_col'): f"{c.upload_time:.2f}s" if c.upload_time else "-",
                t('data_size'): f"{c.upload_bytes/1024:.1f} KB" if c.upload_bytes else "-",
                t('error_col'): c.error_msg or "-"
            })
        
        df = pd.DataFrame(data)
        
        # ★★★ 修复：使用深色主题友好的颜色 ★★★
        def style_status(val):
            if is_completed_status(str(val)):
                # 深绿色背景 + 浅绿色文字
                return 'background-color: #1e4d2b; color: #90EE90'
            elif is_error_status(str(val)):
                # 深红色背景 + 浅红色文字
                return 'background-color: #4d1e1e; color: #ff6b6b'
            elif not is_waiting_status(str(val)) and str(val) not in ["-", "⏳"]:
                # 深蓝色背景 + 浅蓝色文字（进行中状态）
                return 'background-color: #1e3a4d; color: #87CEEB'
            return ''
        
        styled_df = df.style.applymap(style_status, subset=[t('status_col')])
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # 统计记录表
    if tracker and tracker.records:
        st.markdown(f"#### {t('stats_records')}")
        stats_df = tracker.get_dataframe()
        # 本地化列名
        col_map = {
            'client_id': t('id'),
            'training_time_s': t('training_time_col'),
            'upload_time_s': t('upload_time_col'),
            'upload_bytes': t('upload_bytes_col'),
            'upload_KB': t('upload_kb_col'),
            'upload_MB': t('upload_mb_col'),
        }
        stats_df = stats_df.rename(columns=col_map)
        # 本地化汇总行标识
        if t('id') in stats_df.columns:
            stats_df.iloc[:, 0] = stats_df.iloc[:, 0].replace({'TOTAL/AVG': t('total_avg_label')})

        st.dataframe(stats_df, use_container_width=True)
        
        # 下载按钮
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            csv = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                t('download_csv'),
                csv,
                f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            try:
                from io import BytesIO
                output = BytesIO()
                stats_df.to_excel(output, index=False)
                st.download_button(
                    t('download_excel'),
                    output.getvalue(),
                    f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except:
                pass



def render_logs():
    """渲染日志区域"""
    app_state = st.session_state.app_state
    
    st.subheader(t('logs'))
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button(t('clear_logs')):
            app_state.logs = []
            st.rerun()
    
    with col2:
        # 下载日志按钮
        if app_state.logs:
            log_text = "\n".join(app_state.logs)
            st.download_button(
                t('download_logs'),
                log_text.encode('utf-8'),
                f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )
    
    # 日志显示区域
    if app_state.logs:
        # 创建一个固定高度的容器
        log_container = st.container()
        with log_container:
            # 显示最新的日志（最新在上）
            logs_reversed = list(reversed(app_state.logs[-100:]))  # 只显示最新100条
            
            log_html = '<div style="font-family: monospace; font-size: 12px; height: 400px; overflow-y: auto; background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 5px;">'
            
            for log in logs_reversed:
                # 根据日志类型添加颜色
                if "[Error]" in log or "✗" in log:
                    color = "#ff6b6b"
                elif "✓" in log or "完成" in log:
                    color = "#51cf66"
                elif "[Stats]" in log:
                    color = "#74c0fc"
                elif "[TX]" in log or "[RX]" in log:
                    color = "#ffd43b"
                elif "[Python]" in log:
                    color = "#be4bdb"
                elif "[AutoSave]" in log:
                    color = "#ff922b"
                else:
                    color = "#d4d4d4"
                
                # 转义HTML字符
                log_escaped = log.replace("<", "&lt;").replace(">", "&gt;")
                log_html += f'<div style="color: {color}; margin: 2px 0;">{log_escaped}</div>'
            
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info(t('no_logs'))


def render_saved_files():
    """渲染已保存文件列表"""
    output_dir = st.session_state.output_dir
    
    st.subheader(t('saved_files'))
    
    if not os.path.exists(output_dir):
        st.info(t('output_dir_not_exist').format(output_dir))
        return
    
    files = []
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            files.append({
                t('filename'): f,
                t('file_size'): f"{stat.st_size / 1024:.1f} KB",
                t('modified_time'): datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                "路径": fpath
            })
    
    if not files:
        st.info(t('no_saved_files'))
        return
    
    # 按修改时间排序（最新在前）
    files.sort(key=lambda x: x[t('modified_time')], reverse=True)
    
    df = pd.DataFrame(files)
    st.dataframe(df[[t('filename'), t('file_size'), t('modified_time')]], use_container_width=True)
    
    # 提供下载
    st.markdown(f"##### {t('download_files')}")
    selected_file = st.selectbox(t('select_file'), [f[t('filename')] for f in files])
    
    if selected_file:
        fpath = os.path.join(output_dir, selected_file)
        with open(fpath, 'rb') as f:
            st.download_button(
                t('download_file').format(selected_file),
                f.read(),
                selected_file,
                use_container_width=True
            )


# ============================
# 主程序
# ============================
def main():
    render_header()
    
    # 渲染侧边栏
    auto_refresh, refresh_interval, num_clients, is_simulation = render_sidebar()
    
    # 创建选项卡
    tabs = st.tabs([t('tab_progress'), t('tab_charts'), t('tab_data'), t('tab_logs'), t('tab_files')])
    
    with tabs[0]:
        render_metrics()
        st.divider()
        render_progress()
    
    with tabs[1]:
        render_charts()
    
    with tabs[2]:
        render_table()
    
    with tabs[3]:
        render_logs()
    
    with tabs[4]:
        render_saved_files()
    
    # 自动刷新
    if auto_refresh and st.session_state.app_state.is_running:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()