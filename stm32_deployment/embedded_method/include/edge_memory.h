#ifndef EDGE_MEMORY_H
#define EDGE_MEMORY_H

#include <stdlib.h>

/*
 * Memory allocation abstraction used by the embedded method code.
 *
 * The original deployment used an SDRAM heap. For publication, the method code
 * is decoupled from a specific board support package. Define these macros to
 * your platform allocator if needed.
 */

#ifndef EDGE_MALLOC
#define EDGE_MALLOC(size) malloc(size)
#endif

#ifndef EDGE_FREE
#define EDGE_FREE(ptr) free(ptr)
#endif

#ifndef EDGE_HEAP_RESET
#define EDGE_HEAP_RESET() ((void)0)
#endif

/**
 * @brief Allocate memory through the platform allocator.
 * @param size Number of bytes to allocate.
 * @return Pointer to allocated memory, or NULL on failure.
 */
static inline void *edge_malloc(size_t size)
{
    return EDGE_MALLOC(size);
}

/**
 * @brief Free memory through the platform allocator.
 * @param ptr Pointer returned by edge_malloc().
 */
static inline void edge_free(void *ptr)
{
    EDGE_FREE(ptr);
}

/**
 * @brief Reset the platform heap when the platform provides that operation.
 */
static inline void edge_heap_reset(void)
{
    EDGE_HEAP_RESET();
}

#endif
