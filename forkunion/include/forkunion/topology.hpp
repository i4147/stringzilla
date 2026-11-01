/**
 *  @file topology.hpp
 *  @brief The hardware description: memory domains, compute domains, and the topology that holds them.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "capabilities.hpp"
#if FU_ON_LINUX && FU_WITH_OS_THREADS
#include <sys/syscall.h> 
#include <unistd.h>      
#endif
namespace ashvardanian {
namespace forkunion {
/**
 *  @brief The kernel's own identifier for the calling thread, or 0 where there is none.
 *  @sa `pinned_thread_t::id`, which caches it so other threads can read it.
 *
 *  Linux calls it a `pid_t` and hands it out through `gettid`. Darwin has no `gettid` at all, and
 *  spells the same idea `pthread_threadid_np`, returning 64 bits. Both are the number a scheduler
 *  or a profiler will show you; neither is a `pthread_t`.
 */
FU_MAYBE_UNUSED_ static inline std::uint64_t current_thread_id() noexcept {
#if FU_ON_LINUX && FU_WITH_OS_THREADS
    return static_cast<std::uint64_t>(::syscall(SYS_gettid));
#elif FU_ON_APPLE
    std::uint64_t thread_id = 0;
    ::pthread_threadid_np(nullptr, &thread_id);
    return thread_id;
#elif FU_ON_WINDOWS && FU_WITH_OS_THREADS
    return static_cast<std::uint64_t>(::GetCurrentThreadId());
#elif FU_ON_FREEBSD
    return static_cast<std::uint64_t>(::pthread_getthreadid_np());
#else
    return 0;
#endif
}
/**
 *  @brief Names the @b calling thread, which is the only thread every platform lets us name.
 *
 *  Linux's `pthread_setname_np` takes a thread and a name, so a spawner can name its workers. Apple's
 *  takes only a name and always renames the caller. Rather than branch on that at every call, the
 *  worker names itself once it is running - the one shape both kernels agree on.
 */
FU_MAYBE_UNUSED_ static inline void set_current_thread_name(FU_MAYBE_UNUSED_ char const *thread_name) noexcept {
#if FU_ON_LINUX && FU_WITH_OS_THREADS
    (void)::pthread_setname_np(::pthread_self(), thread_name);
#elif FU_ON_APPLE
    (void)::pthread_setname_np(thread_name);
#elif FU_ON_WINDOWS && FU_WITH_OS_THREADS
    using set_thread_description_t = HRESULT(WINAPI *)(HANDLE, PCWSTR);
    HMODULE const kernel32 = ::GetModuleHandleW(L"kernel32.dll");
    if (!kernel32) return;
    auto const set_thread_description =
        reinterpret_cast<set_thread_description_t>(::GetProcAddress(kernel32, "SetThreadDescription"));
    if (!set_thread_description) return;
    wchar_t wide_name[16] = {};
    int const written =
        ::MultiByteToWideChar(CP_UTF8, 0, thread_name, -1, wide_name, static_cast<int>(std::size(wide_name)));
    if (written <= 0) return; 
    wide_name[std::size(wide_name) - 1] = L'\0';
    (void)set_thread_description(::GetCurrentThread(), wide_name);
#endif
}
/**
 *  @brief Upper bound on core IDs this machine may ever report, for sizing masks and names.
 *
 *  Not the same as `hardware_concurrency()` on Linux, where cores can be hot-plugged and the kernel
 *  reserves IDs for cores that are offline right now. Elsewhere the distinction does not exist.
 */
FU_MAYBE_UNUSED_ static inline std::size_t possible_cores() noexcept {
#if FU_ON_WINDOWS
    DWORD const configured = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (configured > 0) return static_cast<std::size_t>(configured);
#elif FU_ON_POSIX
    long const configured = ::sysconf(_SC_NPROCESSORS_CONF);
    if (configured > 0) return static_cast<std::size_t>(configured);
#endif
    return static_cast<std::size_t>(std::thread::hardware_concurrency());
}
#if FU_ON_WINDOWS
static constexpr int win_core_group_shift_k = 6;
static constexpr core_id_t win_core_index_mask_k = (core_id_t {1} << win_core_group_shift_k) - 1;
/** @brief Logical processors per Windows processor group - the `KAFFINITY` bit-width, a hard ABI cap
 *      of 64 @b per @b group, never a cap on total cores (a machine with more uses several groups). */
static constexpr unsigned win_processors_per_group_k = 1u << win_core_group_shift_k;
FU_MAYBE_UNUSED_ static inline core_id_t win_encode_core_id(WORD group, unsigned bit) noexcept {
    return (static_cast<core_id_t>(group) << win_core_group_shift_k) |
           (static_cast<core_id_t>(bit) & win_core_index_mask_k);
}
FU_MAYBE_UNUSED_ static inline WORD win_core_group(core_id_t id) noexcept {
    return static_cast<WORD>(id >> win_core_group_shift_k);
}
FU_MAYBE_UNUSED_ static inline unsigned win_core_index(core_id_t id) noexcept {
    return static_cast<unsigned>(id & win_core_index_mask_k);
}
#endif 
#if FU_ON_WINDOWS
using core_mask_word_t = KAFFINITY;
#elif FU_ON_FREEBSD
using core_mask_word_t = long;
#elif FU_ON_POSIX
using core_mask_word_t = unsigned long;
#else
using core_mask_word_t = std::uint64_t;
#endif
/**
 *  @brief A dense bitset over `core_id_t` - the cores a thread may run on, or should be confined to.
 *
 *  A machine is not the same thing as the slice of it we were handed. `taskset`, a cgroup `cpuset`,
 *  and a batch scheduler all narrow this set, and `hardware_concurrency` sees none of them. Sizing a
 *  pool from the machine and pinning to cores outside the set either escapes the restriction, or -
 *  where the kernel enforces it - crowds every spinning worker onto the few cores that remain.
 *
 *  Every platform hands out a dense core id, so one bitset covers them all: Linux and FreeBSD number
 *  logical processors from zero, and Windows packs `(group << 6) | bit` into the same integer.
 *
 *  @note Not `CPU_ALLOC`. That macro is `malloc` behind a name - glibc's `__sched_cpualloc` rounds
 *      the count up and tail-calls it - which would both bypass this allocator and contradict what
 *      the library promises. A `dynamic_array` sized from `possible_cores()` costs one cold-path
 *      allocation and, unlike a fixed `cpu_set_t`, does not stop at glibc's 1024-core `CPU_SETSIZE`.
 */
template <typename allocator_type_ = std::allocator<core_mask_word_t>>
class core_mask {
    static constexpr std::size_t bits_per_word_k = sizeof(core_mask_word_t) * 8;
    /** @brief Backing words of the bitset, one bit per `core_id_t`. */
    dynamic_array<core_mask_word_t, allocator_type_> words_;
  public:
    core_mask() noexcept = default;
    explicit core_mask(allocator_type_ const &allocator) noexcept : words_(allocator) {}
    /**
     *  @brief Upper bound on the `core_id_t` values this machine can produce - the width a mask covers.
     *  @note Not `possible_cores()`. Windows ids are `(group << 6) | bit`, so a two-group machine with
     *      80 logical processors still emits ids up to 103. Sizing by the core count would drop them.
     */
    static std::size_t id_space() noexcept {
#if FU_ON_WINDOWS
        WORD const groups = ::GetActiveProcessorGroupCount();
        return static_cast<std::size_t>(groups ? groups : 1) * win_processors_per_group_k;
#else
        return possible_cores();
#endif
    }
    /** @retval false on allocation failure, leaving the mask unusable rather than half-sized. */
    bool try_resize_for(std::size_t const cores) noexcept {
        return words_.try_resize(div_ceil(cores, bits_per_word_k));
    }
    /** @brief Sizes the mask to hold every id this machine can produce. @sa `id_space`. */
    bool try_resize() noexcept { return try_resize_for(id_space()); }
    void reset() noexcept { words_.reset(); }
    void clear() noexcept { std::memset(words_.data(), 0, bytes()); }
    bool valid() const noexcept { return !words_.empty(); }
    std::size_t capacity() const noexcept { return words_.size() * bits_per_word_k; }
    std::size_t bytes() const noexcept { return words_.size() * sizeof(core_mask_word_t); }
    void *data() noexcept { return static_cast<void *>(words_.data()); }
    void const *data() const noexcept { return static_cast<void const *>(words_.data()); }
    void add(core_id_t const core) noexcept {
        if (core < 0 || static_cast<std::size_t>(core) >= capacity()) return;
        std::size_t const bit = static_cast<std::size_t>(core);
        words_[bit / bits_per_word_k] |= static_cast<core_mask_word_t>(core_mask_word_t {1} << (bit % bits_per_word_k));
    }
    bool contains(core_id_t const core) const noexcept {
        if (core < 0 || static_cast<std::size_t>(core) >= capacity()) return false;
        std::size_t const bit = static_cast<std::size_t>(core);
        return ((words_[bit / bits_per_word_k] >> (bit % bits_per_word_k)) & core_mask_word_t {1}) != 0;
    }
    std::size_t count() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < words_.size(); ++i)
            total += static_cast<std::size_t>(popcount(static_cast<std::uint64_t>(words_[i])));
        return total;
    }
};
using core_mask_t = core_mask<>;
/**
 *  @brief Reads the cores the calling thread may run on into @p cores.
 *  @retval false where the platform exposes no such mask, which is @b not an error.
 */
FU_MAYBE_UNUSED_ static inline bool try_capture_thread_cores(FU_MAYBE_UNUSED_ core_mask_t &cores) noexcept {
#if FU_ON_LINUX
    std::size_t cores_to_fit = core_mask_t::id_space();
    for (int attempt = 0; attempt < 4; ++attempt, cores_to_fit *= 2) {
        if (!cores.try_resize_for(cores_to_fit)) return false;
        if (::sched_getaffinity(0, cores.bytes(), static_cast<cpu_set_t *>(cores.data())) == 0) return true;
        if (errno != EINVAL) break; 
    }
    cores.reset();
    return false;
#elif FU_ON_WINDOWS
    if (!cores.try_resize()) return false;
    GROUP_AFFINITY affinity = {};
    if (!::GetThreadGroupAffinity(::GetCurrentThread(), &affinity)) return false;
    for (unsigned bit = 0; bit < win_processors_per_group_k; ++bit)
        if (affinity.Mask & (static_cast<KAFFINITY>(1) << bit)) cores.add(win_encode_core_id(affinity.Group, bit));
    return true;
#elif FU_ON_FREEBSD
    if (!cores.try_resize()) return false;
    if (::cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, cores.bytes(),
                             static_cast<cpuset_t *>(cores.data())) == 0)
        return true;
    cores.reset();
    return false;
#else
    return false;
#endif
}
/**
 *  @brief Number of cores the calling thread may run on, or `possible_cores()` where unknowable.
 *  @note Prefer this to `std::thread::hardware_concurrency` when sizing a pool: the latter counts
 *      the machine's cores, not the ones this process was given.
 */
FU_MAYBE_UNUSED_ static inline std::size_t allowed_cores_count() noexcept {
    core_mask_t allowed;
    if (try_capture_thread_cores(allowed)) {
        std::size_t const allowed_count = allowed.count();
        if (allowed_count > 0) return allowed_count;
    }
    return possible_cores();
}
/** @brief The OS thread handle a `colocated_pool` stores, joins, and pins - one per worker. */
#if FU_ON_WINDOWS
using native_thread_t = HANDLE; 
#else
using native_thread_t = pthread_t;
#endif
/**
 *  @brief Confines @p thread to the cores held by @p cores. The one place placement actually happens.
 *  @retval false when the platform exposes no thread placement, which is @b not an error.
 *
 *  Linux hands out a `cpu_set_t` and honours it. FreeBSD spells the same idea `cpuset_t`. Windows
 *  addresses a core by (processor group, bit), packed into each `core_id_t`; a thread lives in exactly
 *  one group, so a mask spanning two is a caller error. Apple Silicon answers `KERN_NOT_SUPPORTED` to
 *  `thread_policy_set` - measured, not assumed - and offers only a Quality-of-Service class, chosen at
 *  creation. So a pool there partitions the @b work by domain and lets the scheduler place the @b threads.
 *  @sa `try_pin_thread_to_cores`, the adaptor that builds a mask from a core list.
 */
FU_MAYBE_UNUSED_ static inline bool try_apply_thread_cores(FU_MAYBE_UNUSED_ native_thread_t thread,
                                                           FU_MAYBE_UNUSED_ core_mask_t const &cores) noexcept {
#if FU_ON_WINDOWS
    GROUP_AFFINITY affinity = {};
    bool group_chosen = false;
    for (std::size_t core = 0; core < cores.capacity(); ++core) {
        core_id_t const id = static_cast<core_id_t>(core);
        if (!cores.contains(id)) continue;
        WORD const group = win_core_group(id);
        if (!group_chosen) affinity.Group = group, group_chosen = true;
        else if (group != affinity.Group)
            return false; 
        affinity.Mask |= static_cast<KAFFINITY>(1) << win_core_index(id);
    }
    return group_chosen && ::SetThreadGroupAffinity(thread, &affinity, nullptr) != 0;
#elif FU_ON_FREEBSD
    if (!cores.valid()) return false;
    return ::pthread_setaffinity_np(thread, cores.bytes(), static_cast<cpuset_t const *>(cores.data())) == 0;
#elif FU_ON_ANDROID
    if (!cores.valid()) return false;
    return ::sched_setaffinity(::pthread_gettid_np(thread), cores.bytes(),
                               static_cast<cpu_set_t const *>(cores.data())) == 0;
#elif FU_WITH_PLACE_THREADS_BY_AFFINITY
    if (!cores.valid()) return false;
    return ::pthread_setaffinity_np(thread, cores.bytes(), static_cast<cpu_set_t const *>(cores.data())) == 0;
#else
    return false; 
#endif
}
/**
 *  @brief Confines @p thread to the @p count cores listed in @p cores.
 *  @retval false when the platform exposes no thread placement, which is @b not an error.
 *  @note A thin adaptor: it builds a `core_mask` and defers to `try_apply_thread_cores`, which is
 *      where the per-platform placement lives. It owns no platform logic of its own.
 */
FU_MAYBE_UNUSED_ static inline bool try_pin_thread_to_cores(FU_MAYBE_UNUSED_ native_thread_t thread,
                                                            FU_MAYBE_UNUSED_ core_id_t const *cores,
                                                            FU_MAYBE_UNUSED_ std::size_t const count) noexcept {
#if FU_WITH_PLACE_THREADS_BY_AFFINITY
    if (count == 0) return false;
    core_mask_t mask;
    if (!mask.try_resize()) return false;
    for (std::size_t i = 0; i < count; ++i) {
        assert(cores[i] >= 0 && "Invalid CPU core ID");
        mask.add(cores[i]);
    }
    return try_apply_thread_cores(thread, mask);
#else
    return false; 
#endif
}
/**
 *  @brief Puts the calling thread back on the cores @p saved recorded before it was pinned.
 *  @note A no-op where nothing was ever narrowed, or where @p saved was never captured.
 *
 *  A pool that narrows the caller owes it the mask it had, not the mask of the whole machine. The
 *  two differ under `taskset`, a cgroup `cpuset`, or any batch scheduler, and widening to the
 *  machine would hand the caller cores this process was never granted.
 *
 *  Note that no NUMA run-node policy is reset here: the pool never sets one. Memory placement goes
 *  through `mbind` on the allocation, not through the calling thread's policy, and `numa_run_on_node`
 *  would rewrite the very CPU mask we just restored.
 */
FU_MAYBE_UNUSED_ static inline bool try_restore_thread_cores(FU_MAYBE_UNUSED_ core_mask_t const &saved) noexcept {
#if FU_WITH_PLACE_THREADS_BY_AFFINITY
    if (!saved.valid()) return false;
#if FU_ON_WINDOWS
    return try_apply_thread_cores(::GetCurrentThread(), saved);
#else
    return try_apply_thread_cores(::pthread_self(), saved);
#endif
#else
    return false;
#endif
}
#if FU_ON_LINUX
/**
 *  @brief Reads one unsigned integer out of a `/sys` or `/proc` file.
 *  @retval false where the file is absent or holds no number - @p value is then untouched.
 */
FU_MAYBE_UNUSED_ static inline bool try_read_uint_at_path(char const *path, std::size_t &value) noexcept {
    FILE *file = ::fopen(path, "r");
    if (!file) return false;
    unsigned long long parsed = 0;
    bool const parsed_one = ::fscanf(file, "%llu", &parsed) == 1;
    ::fclose(file);
    if (parsed_one) value = static_cast<std::size_t>(parsed);
    return parsed_one;
}
/**
 *  @brief Reads the first line of a `/sys` or `/proc` file into @p line, newline and all.
 *  @retval false where the file is absent, empty, or its first line did not fit.
 *  @note Truncation is a failure, not a prefix: a clipped cpulist names fewer cores than the kernel
 *      does, and would read like a complete answer.
 */
FU_MAYBE_UNUSED_ static inline bool try_read_line_at_path(char const *path, char *line,
                                                          std::size_t const line_capacity) noexcept {
    FILE *file = ::fopen(path, "r");
    if (!file) return false;
    bool complete = ::fgets(line, static_cast<int>(line_capacity), file) != nullptr;
    ::fclose(file);
    if (!complete) return false;
    std::size_t const length = std::strlen(line);
    return !(length == line_capacity - 1 && line[line_capacity - 2] != '\n');
}
#endif 
/**
 *  @brief One page size the kernel offers, and how many pages of it exist.
 *
 *  A machine reports several: the base page every allocation uses by default, and whichever huge
 *  page sizes the hardware and kernel agree on. `available_pages` counts what was reserved, and
 *  `free_pages` what nobody has taken yet, so an allocator can tell "unsupported" from "exhausted".
 *  @sa `ram_capabilities`
 */
struct ram_page_setting_t {
    /** @brief Huge page size in bytes, e.g. 4 KB, 2 MB, or 1 GB. */
    std::size_t bytes_per_page {0};
    /** @brief Number of pages available for this size, 0 if not available. */
    std::size_t available_pages {0};
    /** @brief Number of pages available and unused, 0 if not available. */
    std::size_t free_pages {0};
};
static constexpr std::size_t page_size_4k_k = 4ull * 1024ull;                     
static constexpr std::size_t page_size_2m_k = 2ull * 1024ull * 1024ull;           
static constexpr std::size_t page_size_1g_k = 1ull * 1024ull * 1024ull * 1024ull; 
/**
 *  @brief Fetches the RAM page size in bytes.
 *  @retval The size of a memory page in bytes, typically 4096 on most systems.
 *  @note On Linux, this is the system page size, which may differ from Huge Pages sizes.
 */
FU_MAYBE_UNUSED_ static inline std::size_t ram_page_size() noexcept {
#if FU_ON_POSIX
    return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#elif FU_ON_WINDOWS
    SYSTEM_INFO system_info;
    ::GetSystemInfo(&system_info);
    return static_cast<std::size_t>(system_info.dwPageSize);
#else
    return 4096;
#endif
}
/**
 *  @brief Fetches the total RAM amount available on the system in bytes.
 *  @retval Total system RAM in bytes, or 0 if detection fails.
 *  @note This function provides cross-platform detection of total physical memory.
 */
FU_MAYBE_UNUSED_ static inline std::size_t volume_ram() noexcept {
#if FU_ON_LINUX
    FILE *meminfo_file = ::fopen("/proc/meminfo", "r");
    if (!meminfo_file) return 0;
    char line[256];
    while (::fgets(line, sizeof(line), meminfo_file)) {
        if (::strncmp(line, "MemTotal:", 9) == 0) {
            std::size_t memory_kb = 0;
            if (::sscanf(line, "MemTotal: %zu kB", &memory_kb) == 1) {
                ::fclose(meminfo_file);
                return memory_kb * 1024; 
            }
        }
    }
    ::fclose(meminfo_file);
    return 0;
#elif FU_ON_APPLE
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    std::uint64_t memory_bytes = 0;
    std::size_t size = sizeof(memory_bytes);
    if (::sysctl(mib, 2, &memory_bytes, &size, nullptr, 0) == 0) return static_cast<std::size_t>(memory_bytes);
    return 0;
#elif FU_ON_WINDOWS
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (::GlobalMemoryStatusEx(&mem_status)) return static_cast<std::size_t>(mem_status.ullTotalPhys);
    return 0;
#elif FU_ON_POSIX
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
    return 0;
#else
    return 0;
#endif
}
/**
 *  @brief Describes the configured & supported (by OS & CPU) memory pages sizes.
 *
 *  This class avoids HugeTLBfs in favor of a direct access to the @b `/sys` filesystem.
 *  Aside from fetching the stats, it also allows us to change settings if admin privileges
 *  are granted to running process.
 *
 *  @section Huge Pages & Transparent Huge Pages
 *
 *  Virtual Address Space (VAS) is divided into pages, typically 4 KB in size.
 *  Converting a virtual address to a physical address requires a page table lookup.
 *  Think of it as a hash table... and as everyone knows, hash table lookups and updates
 *  aren't free, so most chips have a "Translation Lookaside Buffer" @b (TLB) cache
 *  as part of the "Memory Management Unit" @b (MMU) to speed up the process.
 *
 *  To keep it fast, in Big Data applications, one would like to use larger pages,
 *  to reduce the number of distinct entries in the TLB cache. Going from 4 KB to
 *  2 MB or 1 GB "Huge Pages" @b (HPs), reduces the table size by 512 or 262K times,
 *  respectively.
 *
 *  To benefit from those, some applications rely on "Transparent Huge Pages" @b (THP),
 *  which are automatically allocated by the kernel. Such implicit behaviour isn't
 *  great for performance-oriented applications, so this inventory exposes @b `largest_free`, which
 *  the `linux_numa_allocator` consults before falling back to the base page size.
 *
 *  @see https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
 */
template <std::size_t max_page_sizes_ = 4>
class ram_page_settings {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;
    /** @brief Huge page sizes in bytes; a machine offers a handful, so the storage is inline. */
    limited_array<ram_page_setting_t, max_page_sizes_k> sizes_ {};
    /** @brief Total memory available on this memory domain. */
    std::size_t total_memory_bytes_ {0};
  public:
    /**
     *  @brief Finds the largest Huge Pages size available for the given NUMA node.
     */
    ram_page_setting_t largest_free() const noexcept {
        if (sizes_.empty()) return {};
        ram_page_setting_t largest = sizes_[0];
        for (std::size_t i = 1; i < sizes_.size(); ++i)
            if (sizes_[i].free_pages > largest.free_pages) largest = sizes_[i];
        return largest;
    }
    std::size_t size() const noexcept { return sizes_.size(); }
    std::size_t total_memory_bytes() const noexcept { return total_memory_bytes_; }
    ram_page_setting_t const *begin() const noexcept { return sizes_.data(); }
    ram_page_setting_t const *end() const noexcept { return sizes_.data() + sizes_.size(); }
    ram_page_setting_t const &operator[](std::size_t const index) const noexcept {
        assert(index < sizes_.size() && "Index is out of bounds");
        return sizes_[index];
    }
    /**
     *  @brief Fetches all available huge page sizes for the given NUMA node.
     *  @note Kernel support doesn't mean that pages of that size have a valid mount point.
     */
    bool try_harvest(FU_MAYBE_UNUSED_ memory_domain_id_t memory_domain_id) noexcept {
        assert(memory_domain_id >= 0 && "NUMA node ID must be non-negative");
#if FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN && FU_ON_LINUX
        sizes_.clear();
        char hugepages_path[256];
        int path_result = std::snprintf(            
            hugepages_path, sizeof(hugepages_path), 
            "/sys/devices/system/node/node%d/hugepages", memory_domain_id);
        if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(hugepages_path))
            return false; 
        DIR *hugepages_dir = ::opendir(hugepages_path);
        if (!hugepages_dir) return false; 
        struct dirent *entry;
        while ((entry = ::readdir(hugepages_dir)) != nullptr && !sizes_.full()) {
            if (entry->d_type != DT_DIR) continue;
            if (::strncmp(entry->d_name, "hugepages-", 10) != 0) continue;
            char const *size_start = entry->d_name + 10; 
            char *size_end = nullptr;
            std::size_t bytes_per_page_kb = static_cast<std::size_t>(::strtoull(size_start, &size_end, 10));
            if (!size_end || std::strcmp(size_end, "kB") != 0) continue;
            if (bytes_per_page_kb == 0) continue; 
            std::size_t const bytes_per_page = bytes_per_page_kb * 1024;
            char nr_hugepages_path[512];
            char free_hugepages_path[512];
            path_result = std::snprintf(                      
                nr_hugepages_path, sizeof(nr_hugepages_path), 
                "%s/%s/nr_hugepages", hugepages_path, entry->d_name);
            if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(nr_hugepages_path))
                continue; 
            path_result = std::snprintf(                          
                free_hugepages_path, sizeof(free_hugepages_path), 
                "%s/%s/free_hugepages", hugepages_path, entry->d_name);
            if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(free_hugepages_path))
                continue; 
            FILE *nr_file = ::fopen(nr_hugepages_path, "r");
            if (!nr_file) continue; 
            std::size_t allocated_pages = 0;
            std::size_t free_pages = 0;
            if (::fscanf(nr_file, "%zu", &allocated_pages) != 1) {
                ::fclose(nr_file);
                continue; 
            }
            ::fclose(nr_file);
            FILE *free_file = ::fopen(free_hugepages_path, "r");
            if (free_file) {
                if (::fscanf(free_file, "%zu", &free_pages) != 1) {
                    free_pages = 0; 
                }
                ::fclose(free_file);
            }
            ram_page_setting_t setting {};
            setting.bytes_per_page = bytes_per_page;
            setting.available_pages = allocated_pages;
            setting.free_pages = free_pages;
            sizes_.try_push_back(setting); 
        }
        ::closedir(hugepages_dir);
        char meminfo_path[256];
        path_result = std::snprintf(meminfo_path, sizeof(meminfo_path), "/sys/devices/system/node/node%d/meminfo",
                                    memory_domain_id);
        if (path_result > 0 && static_cast<std::size_t>(path_result) < sizeof(meminfo_path)) {
            FILE *meminfo_file = ::fopen(meminfo_path, "r");
            if (meminfo_file) {
                char line[256];
                while (::fgets(line, sizeof(line), meminfo_file)) {
                    if (::strncmp(line, "Node ", 5) == 0 && ::strstr(line, " MemTotal:")) {
                        std::size_t memory_kb = 0;
                        if (::sscanf(line, "Node %*d MemTotal: %zu kB", &memory_kb) == 1) {
                            total_memory_bytes_ = memory_kb * 1024; 
                            break;
                        }
                    }
                }
                ::fclose(meminfo_file);
            }
        }
        return true;
#elif FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN && FU_ON_WINDOWS
        fu_unused_(memory_domain_id);
        SIZE_T const large_page_bytes = ::GetLargePageMinimum();
        if (large_page_bytes == 0) return false; 
        ram_page_setting_t only {};
        only.bytes_per_page = static_cast<std::size_t>(large_page_bytes);
        only.available_pages = 0;
        only.free_pages = 0;
        sizes_.clear();
        sizes_.try_push_back(only);
        total_memory_bytes_ = 0;
        return true;
#else
        fu_unused_(memory_domain_id);
        return false;
#endif
    }
    /**
     *  @brief Attempts to reserve huge pages of a specific size on the current NUMA node.
     *  @param[in] page_size_bytes The size of huge pages to reserve (must match an available size)
     *  @param[in] num_pages Number of pages to reserve
     *  @return true if reservation was successful, false otherwise
     *  @note Requires root privileges or appropriate capabilities
     */
    bool try_change(memory_domain_id_t memory_domain_id, std::size_t page_size_bytes, std::size_t num_pages) noexcept {
        assert(memory_domain_id >= 0 && "NUMA node ID must be non-negative");
        std::size_t page_index = sizes_.size();
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            if (sizes_[i].bytes_per_page == page_size_bytes) {
                page_index = i;
                break;
            }
        }
        if (page_index >= sizes_.size()) return false; 
        std::size_t const page_size_kb = page_size_bytes / 1024;
        char nr_hugepages_path[512];
        int const path_result = std::snprintf(                                        
            nr_hugepages_path, sizeof(nr_hugepages_path),                             
            "/sys/devices/system/node/node%d/hugepages/hugepages-%zukB/nr_hugepages", 
            memory_domain_id, page_size_kb);
        if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(nr_hugepages_path))
            return false; 
        FILE *nr_file = ::fopen(nr_hugepages_path, "w");
        if (!nr_file) return false; 
        bool const update_success = (::fprintf(nr_file, "%zu", num_pages) > 0);
        ::fclose(nr_file);
        if (!update_success) return false; 
        return try_harvest(memory_domain_id);
    }
};
using ram_page_settings_t = ram_page_settings<>;
/**
 *  @brief Describes a NUMA node, containing its ID, memory size, and core IDs.
 *  @sa Views different slices of the `machine_topology` structure.
 */
template <std::size_t max_page_sizes_ = 4>
struct memory_domain {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;
    /** @brief The OS's id for this memory domain: a NUMA node number, in [0, numa_max_node()]. */
    memory_domain_id_t memory_domain_id {-1};
    /** @brief Physical CPU socket ID. */
    socket_id_t socket_id {-1};
    /** @brief RAM volume in bytes. */
    std::size_t volume_ram {0};
    /** @brief Pointer to the first core ID in the `core_ids` array. */
    core_id_t const *first_core_id {nullptr};
    /** @brief Number of items in the `core_ids` array. */
    std::size_t logical_cores_count {0};
    /** @brief Huge page sizes available on this memory domain. */
    ram_page_settings<max_page_sizes_k> page_sizes {};
};
using memory_domain_t = memory_domain<>;
/**
 *  @brief One bindable cluster of cores sharing a QoS class and locality - the compute axis.
 *  @sa `memory_domain` is the memory axis; a `machine_topology` exposes both plus their affinity.
 *
 *  A compute domain is a contiguous run of same-capacity cores within a single NUMA node. It is
 *  what a pool binds to and the index a worker callback receives. Several compute domains may map
 *  to one memory domain (performance and efficiency cores sharing a memory controller), which is
 *  why compute and memory are separate axes rather than a single "colocation" cell.
 */
struct compute_domain_t {
    /** @brief The OS's id for the memory domain these cores live on. */
    memory_domain_id_t memory_domain_id {-1};
    /** @brief Our dense index for that same memory domain, into `machine topology`'s array. */
    memory_domain_index_t memory_domain_index {};
    /** @brief QoS ordinal, sorted least-to-most performant. */
    std::size_t compute_level {0};
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
    /**
     *  @brief Apple's absolute class for these cores, from `hw.perflevelN.name`; -1 when unnamed.
     *  @sa `colocated_pool`'s `_qos_for_domain`, for why the relative rank above cannot replace it.
     *
     *  Spawn-path plumbing, not a second caller-facing rank: like `first_core_id`, it is the raw
     *  material the platform's placement call consumes - an affinity mask there, a QoS class here.
     */
    core_quality_t apple_core_quality {-1};
#endif
    /**
     *  @brief Relative throughput of @b one core here; 0 when the platform exposes no rating.
     *  @sa `compute_level` is a dense ordinal for grouping - never divide by it.
     *
     *  Unlike `compute_level`, this is a magnitude, so it may be summed and divided. Where the
     *  kernel publishes a per-core rating - such as the Linux scheduler's `cpu_capacity`, scaled so
     *  the fastest core present reads 1024 - it lands here. Platforms that rank cores without
     *  quantifying them leave this unknown, and callers weigh domains by `logical_cores_count` instead.
     */
    std::size_t capacity {0};
    /**
     *  @brief Bytes of the deepest cache private to this domain's cores; 0 when unknown.
     *
     *  Sizes a cache-resident chunk, which is a different question from how many chunks a domain
     *  should receive - cores of equal throughput may back onto very differently sized caches, so
     *  neither number can be derived from the other. Shared by every core in the domain.
     */
    std::size_t cache_bytes {0};
    /** @brief Pointer to the first core ID in this domain. */
    core_id_t const *first_core_id {nullptr};
    /** @brief Number of cores in this domain. */
    std::size_t logical_cores_count {0};
};
/**
 *  @brief Fetches the socket ID for a given CPU core.
 *  @param[in] core_id The CPU core ID to query.
 *  @retval Socket ID (>= 0) if successful.
 *  @retval -1 if failed.
 */
FU_MAYBE_UNUSED_ static inline socket_id_t socket_id_of_core(FU_MAYBE_UNUSED_ core_id_t core_id) noexcept {
    int socket_id = -1;
#if FU_ON_LINUX
    char socket_path[256];
    int path_result = std::snprintf(      
        socket_path, sizeof(socket_path), 
        "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", core_id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(socket_path)) return -1; 
    FILE *socket_file = ::fopen(socket_path, "r");
    if (!socket_file) return -1; 
    if (::fscanf(socket_file, "%d", &socket_id) != 1) socket_id = -1; 
    ::fclose(socket_file);
#endif
    return socket_id;
}
/**
 *  @brief Fetches the scheduler capacity of a CPU core, used to separate QoS classes.
 *  @retval A capacity value where larger means more performant, or 0 if unavailable.
 *
 *  Reads `/sys/devices/system/cpu/cpuN/cpu_capacity`, which the kernel populates from the
 *  Energy Model on ARM @b big.LITTLE/DynamIQ and from ITMT/Thread-Director on Intel @b hybrid
 *  chips (performance cores report ~1024, efficiency cores less). A return of 0 means the core
 *  is homogeneous or the kernel does not expose capacities - the whole node is then one class.
 */
FU_MAYBE_UNUSED_ static inline std::size_t capacity_of_core(FU_MAYBE_UNUSED_ core_id_t core_id) noexcept {
    std::size_t capacity = 0;
#if FU_ON_LINUX
    char capacity_path[256];
    int path_result = std::snprintf(          
        capacity_path, sizeof(capacity_path), 
        "/sys/devices/system/cpu/cpu%d/cpu_capacity", core_id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(capacity_path)) return 0; 
    FILE *capacity_file = ::fopen(capacity_path, "r");
    if (!capacity_file) return 0; 
    unsigned long long parsed = 0;
    if (::fscanf(capacity_file, "%llu", &parsed) == 1) capacity = static_cast<std::size_t>(parsed);
    ::fclose(capacity_file);
#endif
    return capacity;
}
/**
 *  @brief Hands every `[low, high]` range of a Linux id list - "0", "0-3", or "0,2-4" - to @p visit.
 *  @note One parser for every list the kernel publishes: `shared_cpu_list`, a domain's `cpulist`, and
 *      `node/online`. Ranges arrive inclusive on both ends, as written.
 */
template <typename visitor_type_>
FU_MAYBE_UNUSED_ static inline void for_each_id_list_range(char const *line, visitor_type_ &&visit) noexcept {
    for (char const *cursor = line; *cursor;) {
        char *next = nullptr;
        long const low = ::strtol(cursor, &next, 10);
        if (next == cursor) break; 
        long high = low;
        cursor = next;
        if (*cursor == '-') high = ::strtol(cursor + 1, &next, 10), cursor = next;
        visit(low, high);
        while (*cursor == ',' || *cursor == ' ' || *cursor == '\n') ++cursor;
    }
}
/**
 *  @brief Whether every core in a Linux cpulist line - "0", "0-3", or "0,2-4" - belongs to @p cores.
 *  @note Pass only complete lines: a truncated tail could name cores the verdict never saw.
 */
FU_MAYBE_UNUSED_ static inline bool cpu_list_within(char const *line, core_id_t const *cores,
                                                    std::size_t const cores_count) noexcept {
    bool within = true;
    for_each_id_list_range(line, [&](long const low, long const high) noexcept {
        for (long listed = low; listed <= high && within; ++listed) {
            bool found = false;
            for (std::size_t i = 0; i != cores_count && !found; ++i) found = cores[i] == static_cast<core_id_t>(listed);
            if (!found) within = false;
        }
    });
    return within;
}
/**
 *  @brief Bytes of the deepest data or unified cache serving @p core_id, confined to @p domain_cores.
 *  @retval The cache size in bytes, or 0 when no platform source can name it.
 *
 *  Two exact sources, no measurement: Linux's per-core cacheinfo sysfs, counting a level only if
 *  its `shared_cpu_list` stays within the domain - a socket-wide L3 is not one QoS class's to
 *  claim; elsewhere x86 CPUID leaf 0x4 / 0x8000001D, with the sharing width standing in for that
 *  containment. Arm has no userspace cache-geometry registers (`CCSIDR_EL1` is EL1-only), so
 *  sysfs is the only Arm source and other Arm hosts honestly report 0.
 */
FU_MAYBE_UNUSED_ static inline std::size_t cache_bytes_of_core(
    FU_MAYBE_UNUSED_ core_id_t core_id, FU_MAYBE_UNUSED_ core_id_t const *domain_cores,
    FU_MAYBE_UNUSED_ std::size_t domain_cores_count) noexcept {
#if FU_ON_LINUX
    std::size_t deepest_bytes = 0;
    for (int index = 0; index < 16; ++index) {
        char path[256];
        int written = std::snprintf(path, sizeof(path), 
                                    "/sys/devices/system/cpu/cpu%d/cache/index%d/type", core_id, index);
        if (written < 0 || static_cast<std::size_t>(written) >= sizeof(path)) break; 
        FILE *type_file = ::fopen(path, "r");
        if (!type_file) break; 
        char type_name[16] = {0};
        bool const parsed_type = ::fscanf(type_file, "%15s", type_name) == 1;
        ::fclose(type_file);
        if (!parsed_type || type_name[0] == 'I') continue; 
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", core_id,
                      index);
        FILE *shared_file = ::fopen(path, "r");
        if (!shared_file) continue;
        char line[256] = {0};
        bool complete = ::fgets(line, sizeof(line), shared_file) != nullptr;
        ::fclose(shared_file);
        complete = complete && !(std::strlen(line) == sizeof(line) - 1 && line[sizeof(line) - 2] != '\n');
        if (!complete || !cpu_list_within(line, domain_cores, domain_cores_count)) continue;
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/size", core_id, index);
        FILE *size_file = ::fopen(path, "r");
        if (!size_file) continue;
        unsigned long long parsed = 0;
        char suffix = 0;
        int const fields = ::fscanf(size_file, "%llu%c", &parsed, &suffix);
        ::fclose(size_file);
        if (fields < 1) continue;
        std::size_t bytes = static_cast<std::size_t>(parsed);
        if (fields == 2 && (suffix == 'K' || suffix == 'k')) bytes <<= 10;
        if (fields == 2 && (suffix == 'M' || suffix == 'm')) bytes <<= 20;
        if (fields == 2 && (suffix == 'G' || suffix == 'g')) bytes <<= 30;
        if (bytes > deepest_bytes) deepest_bytes = bytes;
    }
    if (deepest_bytes) return deepest_bytes;
#endif
#if FU_DETECT_ARCH_X86_64_
    std::uint32_t const deterministic_leaf = 0x8000'001Du;
    std::uint32_t const max_extended = cpuid(0x8000'0000u, 0).eax;
    if (max_extended < deterministic_leaf && cpuid(0, 0).eax < 0x4u) {
        if (max_extended < 0x8000'0006u) return 0;
        return static_cast<std::size_t>(cpuid(0x8000'0006u, 0).ecx >> 16) << 10;
    }
    std::uint32_t const leaf = max_extended >= deterministic_leaf ? deterministic_leaf : 0x4u;
    std::size_t deepest_cpuid_bytes = 0;
    for (std::uint32_t subleaf = 0; subleaf < 16; ++subleaf) {
        cpuid_registers_t const r = cpuid(leaf, subleaf);
        std::uint32_t const cache_type = r.eax & 0x1Fu;
        if (cache_type == 0) break;
        if (cache_type == 2) continue;
        std::size_t const sharing_cores = ((r.eax >> 14) & 0xFFFu) + 1;
        if (sharing_cores > domain_cores_count) continue;
        std::size_t const line_bytes = (r.ebx & 0xFFFu) + 1;
        std::size_t const partitions = ((r.ebx >> 12) & 0x3FFu) + 1;
        std::size_t const ways = ((r.ebx >> 22) & 0x3FFu) + 1;
        std::size_t const sets = static_cast<std::size_t>(r.ecx) + 1;
        std::size_t const bytes = line_bytes * partitions * ways * sets;
        if (bytes > deepest_cpuid_bytes) deepest_cpuid_bytes = bytes;
    }
    return deepest_cpuid_bytes;
#else
    return 0;
#endif
}
#if FU_WITH_TOPOLOGY && FU_ON_LINUX
static constexpr char const *sysfs_node_root_k = "/sys/devices/system/node";
/**
 *  @brief Whether this kernel enumerates memory domains at all - what `numa_available` answered.
 *  @note A kernel built without `CONFIG_NUMA` mounts no such directory.
 */
FU_MAYBE_UNUSED_ static inline bool linux_has_memory_domains() noexcept {
    DIR *node_dir = ::opendir(sysfs_node_root_k);
    if (!node_dir) return false;
    ::closedir(node_dir);
    return true;
}
/**
 *  @brief Highest online memory-domain id, or -1 where none can be read - what `numa_max_node` gave.
 *  @note `node/online` is an id list, and hot-unplug makes it gappy ("0,2"), so this is a ceiling only.
 */
FU_MAYBE_UNUSED_ static inline memory_domain_id_t max_memory_domain_id() noexcept {
    char path[256], line[256];
    int const path_result = std::snprintf(path, sizeof(path), "%s/online", sysfs_node_root_k);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(path)) return -1; 
    if (!try_read_line_at_path(path, line, sizeof(line))) return -1;
    long highest = -1;
    for_each_id_list_range(line, [&](long, long const high) noexcept {
        if (high > highest) highest = high;
    });
    return static_cast<memory_domain_id_t>(highest);
}
/**
 *  @brief Reads one memory domain's total RAM into @p bytes - what `numa_node_size64` returned.
 *  @retval false where the domain publishes no `meminfo` - offline, or absent from a gappy id space.
 *  @note Fallible rather than 0-sentinel because the distinction is load-bearing: false is that
 *      function's negative return, while true with zero @p bytes is a memoryless domain, still real.
 *      Its `free` out-parameter is not mirrored - the only caller wrote it and never read it.
 */
FU_MAYBE_UNUSED_ static inline bool try_read_ram_bytes_of_memory_domain(memory_domain_id_t const id,
                                                                        std::size_t &bytes) noexcept {
    char path[256], line[256];
    int const path_result = std::snprintf(path, sizeof(path), "%s/node%d/meminfo", sysfs_node_root_k, id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(path)) return false; 
    FILE *meminfo_file = ::fopen(path, "r");
    if (!meminfo_file) return false; 
    bytes = 0;
    while (::fgets(line, sizeof(line), meminfo_file)) {
        std::size_t memory_kb = 0;
        if (::strncmp(line, "Node ", 5) == 0 && ::strstr(line, " MemTotal:") &&
            ::sscanf(line, "Node %*d MemTotal: %zu kB", &memory_kb) == 1) {
            bytes = memory_kb * 1024;
            break;
        }
    }
    ::fclose(meminfo_file);
    return true;
}
/**
 *  @brief Reads the cores of one memory domain into @p cores - what `numa_node_to_cpus` filled.
 *  @retval false where the domain names no cpulist, or the mask could not be sized to hold it.
 *  @note A `core_mask`, so the harvest's own allocator owns it - `numa_allocate_cpumask` malloc'd
 *      behind its back, the very thing `core_mask` refuses `CPU_ALLOC` over.
 */
FU_MAYBE_UNUSED_ static inline bool try_capture_memory_domain_cores(memory_domain_id_t const id,
                                                                    core_mask_t &cores) noexcept {
    char path[256], line[1024];
    int const path_result = std::snprintf(path, sizeof(path), "%s/node%d/cpulist", sysfs_node_root_k, id);
    if (path_result < 0 || static_cast<std::size_t>(path_result) >= sizeof(path)) return false; 
    if (!try_read_line_at_path(path, line, sizeof(line))) return false;
    if (!cores.try_resize()) return false; 
    cores.clear();
    for_each_id_list_range(line, [&](long const low, long const high) noexcept {
        for (long listed = low; listed <= high; ++listed) cores.add(static_cast<core_id_t>(listed));
    });
    return true;
}
#endif 
#if FU_ON_WINDOWS
/**
 *  @brief One accumulator per (processor group, efficiency class): the union of that class's core masks
 *      within the group, and the largest private cache seen for it.
 *  @sa `try_harvest_windows` builds these from the processor-core and cache relationships, then reads
 *      them back per NUMA node - so it never keeps a per-processor scratch table.
 */
struct win_group_class_cell_t {
    /** @brief OR of every core of this class in this group. */
    KAFFINITY mask {0};
    /** @brief Largest L1/L2 (private) cache seen for those cores. */
    std::size_t cache_bytes {0};
};
/**
 *  @brief Detects whether this SDK's `NUMA_NODE_RELATIONSHIP` exposes the multi-group `GroupMasks[]`.
 *  @note Version macros are unreliable here - MinGW reports Windows 7 yet defines the member - so we
 *      probe the member itself. Older SDKs model a node as a single group, read by the fallback.
 */
template <typename numa_relationship_type_, typename = void>
struct win_numa_has_group_masks : std::false_type {};
template <typename numa_relationship_type_>
struct win_numa_has_group_masks<numa_relationship_type_,
                                std::void_t<decltype(std::declval<numa_relationship_type_ &>().GroupMasks)>>
    : std::true_type {};
/**
 *  @brief Invokes @p visitor(group, mask) for every processor group a NUMA @p node owns.
 *  @note Templated on the node type so the discarded `if constexpr` branch is dependent and only the
 *      supported member is ever compiled. A node spanning several groups (the largest servers) is
 *      thus enumerated in full on new SDKs, and read through its single group on old ones.
 */
template <typename numa_relationship_type_, typename visitor_type_>
static inline void win_numa_for_each_group(numa_relationship_type_ const &node, visitor_type_ &&visitor) noexcept {
    if constexpr (win_numa_has_group_masks<numa_relationship_type_>::value) {
        if (node.GroupCount == 0) return visitor(node.GroupMask.Group, node.GroupMask.Mask);
        for (WORD g = 0; g < node.GroupCount; ++g) visitor(node.GroupMasks[g].Group, node.GroupMasks[g].Mask);
    }
    else { visitor(node.GroupMask.Group, node.GroupMask.Mask); }
}
/**
 *  @brief Best-effort socket id for a NUMA @p node: the processor package that owns its cores.
 *  @retval @p fallback when package data is unavailable or no package matches.
 *  @note Packages and nodes are both few, so scanning the package buffer per node needs no scratch.
 */
FU_MAYBE_UNUSED_ static inline socket_id_t win_socket_for_node( 
    BYTE const *package_buffer, DWORD package_len, NUMA_NODE_RELATIONSHIP const &node, socket_id_t fallback) noexcept {
    if (!package_buffer) return fallback;
    socket_id_t socket_index = 0;
    for (DWORD offset = 0; offset < package_len;) {
        auto const *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX const *>(package_buffer + offset);
        if (record->Relationship == RelationProcessorPackage) {
            bool intersects = false;
            PROCESSOR_RELATIONSHIP const &package = record->Processor;
            for (WORD g = 0; g < package.GroupCount; ++g) {
                GROUP_AFFINITY const package_affinity = package.GroupMask[g];
                win_numa_for_each_group(node, [&](WORD node_group, KAFFINITY node_mask) noexcept {
                    if (node_group == package_affinity.Group && (node_mask & package_affinity.Mask)) intersects = true;
                });
            }
            if (intersects) return socket_index;
            socket_index += 1;
        }
        if (record->Size == 0) break;
        offset += record->Size;
    }
    return fallback;
}
#endif 
#if FU_ON_APPLE
/**
 *  @brief Reads an unsigned integer `sysctl` by name (e.g. "hw.nperflevels"), or 0 if unavailable.
 *  @sa Used to harvest the Apple Silicon performance-level topology.
 */
FU_MAYBE_UNUSED_ static inline std::size_t apple_sysctl_uint(char const *name) noexcept {
    unsigned long long value = 0;
    std::size_t length = sizeof(value);
    if (::sysctlbyname(name, &value, &length, nullptr, 0) != 0) return 0;
    return static_cast<std::size_t>(value);
}
#endif 
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
/** @brief Reads a string `sysctl` by name into @p out (always NUL-terminated), returning success. */
FU_MAYBE_UNUSED_ static inline bool apple_sysctl_string(char const *name, char *out, std::size_t cap) noexcept {
    if (cap == 0) return false;
    std::size_t length = cap;
    if (::sysctlbyname(name, out, &length, nullptr, 0) != 0) {
        out[0] = '\0';
        return false;
    }
    out[cap - 1] = '\0';
    return true;
}
/**
 *  @brief Apple's `hw.perflevelN.name` vocabulary as an @b absolute ladder; higher is more performant.
 *  @note Absolute, unlike `compute_level`: "Performance" is the same class on an M1 (its top tier)
 *      and an M5 Pro (its bottom). Apple has shipped exactly these three names.
 */
enum apple_core_quality_t : core_quality_t {
    apple_efficiency_k = 0,  
    apple_performance_k = 1, 
    apple_super_k = 2,       
};
/**
 *  @brief The name Apple gives an absolute core class, or `nullptr` for an unknown one.
 *  @note The one source of truth for the names; `apple_core_quality_from_name` inverts it.
 */
FU_MAYBE_UNUSED_ static inline char const *apple_core_quality_name(core_quality_t const quality) noexcept {
    switch (quality) {
    case apple_efficiency_k: return "Efficiency";
    case apple_performance_k: return "Performance";
    case apple_super_k: return "Super";
    default: return nullptr;
    }
}
/**
 *  @brief Maps a `hw.perflevelN.name` to its absolute class; inverts `apple_core_quality_name`.
 *  @retval apple_performance_k for a null or unrecognised name, so `UTILITY` is never guessed.
 */
FU_MAYBE_UNUSED_ static inline core_quality_t apple_core_quality_from_name(char const *name) noexcept {
    if (name == nullptr) return apple_performance_k;
    for (core_quality_t quality = apple_efficiency_k; quality <= apple_super_k; ++quality) {
        char const *const candidate = apple_core_quality_name(quality);
        if (candidate != nullptr && std::strcmp(candidate, name) == 0) return quality;
    }
    return apple_performance_k;
}
#endif 
/**
 *  @brief NUMA topology descriptor: describing memory pools and core counts next to them.
 *
 *  Uses dynamic memory to store the NUMA nodes and their cores. Assuming we may soon have
 *  Intel "Sierra Forest"-like CPUs with 288 cores with up to 8 sockets per node, this structure
 *  can easily grow to 10 KB.
 */
template <std::size_t max_page_sizes_ = 4, typename allocator_type_ = std::allocator<char>>
struct machine_topology {
    using allocator_t = allocator_type_;
    using cores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<int>;
    using memory_domains_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<memory_domain_t>;
    using domains_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<compute_domain_t>;
    using capacities_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::size_t>;
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;
  private:
    /** @brief Allocator that backs every heap array below. */
    allocator_t allocator_ {};
    /** @brief Memory domains, one per NUMA node. */
    dynamic_array<memory_domain_t, memory_domains_allocator_t> memory_domains_;
    /** @brief Core IDs grouped by node then QoS; the nodes and domains below slice into this. */
    dynamic_array<core_id_t, cores_allocator_t> domain_core_ids_;
    /** @brief Compute domains, one per same-QoS core run within a node; sized for the worst case. */
    dynamic_array<compute_domain_t, domains_allocator_t> compute_domains_;
    /** @brief Number of memory domains, one per NUMA node. */
    std::size_t memory_domains_count_ {0};
    /** @brief Total number of cores in all nodes. */
    std::size_t logical_cores_count_ {0};
    /** @brief Number of compute domains actually written, never more than `compute_domains_.size()`. */
    std::size_t compute_domains_count_ {0};
    /** @brief Number of distinct QoS classes (>= 1). */
    std::size_t compute_levels_count_ {1};
  public:
    constexpr machine_topology() noexcept = default;
    machine_topology(machine_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), memory_domains_(std::move(o.memory_domains_)),
          domain_core_ids_(std::move(o.domain_core_ids_)), compute_domains_(std::move(o.compute_domains_)),
          memory_domains_count_(std::exchange(o.memory_domains_count_, 0)),
          logical_cores_count_(std::exchange(o.logical_cores_count_, 0)),
          compute_domains_count_(std::exchange(o.compute_domains_count_, 0)),
          compute_levels_count_(std::exchange(o.compute_levels_count_, 1)) {}
    machine_topology &operator=(machine_topology &&other) noexcept {
        if (this != &other) {
            allocator_ = std::move(other.allocator_);
            memory_domains_ = std::move(other.memory_domains_);
            domain_core_ids_ = std::move(other.domain_core_ids_);
            compute_domains_ = std::move(other.compute_domains_);
            memory_domains_count_ = std::exchange(other.memory_domains_count_, 0);
            logical_cores_count_ = std::exchange(other.logical_cores_count_, 0);
            compute_domains_count_ = std::exchange(other.compute_domains_count_, 0);
            compute_levels_count_ = std::exchange(other.compute_levels_count_, 1);
        }
        return *this;
    }
    machine_topology(machine_topology const &) = delete;
    machine_topology &operator=(machine_topology const &) = delete;
    ~machine_topology() noexcept { reset(); }
    void reset() noexcept {
        memory_domains_.reset();
        domain_core_ids_.reset();
        compute_domains_.reset();
        memory_domains_count_ = logical_cores_count_ = compute_domains_count_ = 0;
        compute_levels_count_ = 1;
    }
    /**
     *  @brief Copy-assigns the topology from @p other.
     *
     *  Instead of a copy-constructor we expose an explicit operation that can
     *  FAIL - returning `false` if *any* intermediate allocation fails.
     *
     *  @param other Source topology.
     *  @retval true  Success, the current instance now owns a deep copy.
     *  @retval false Allocation failed, the current instance is unchanged.
     */
    bool try_assign(machine_topology const &other) noexcept {
        if (this == &other) return true; 
        dynamic_array<memory_domain_t, memory_domains_allocator_t> scratch_nodes {
            memory_domains_allocator_t {allocator_}};
        dynamic_array<core_id_t, cores_allocator_t> scratch_core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> scratch_domains {domains_allocator_t {allocator_}};
        if (!scratch_nodes.try_resize(other.memory_domains_count_)) return false;   
        if (!scratch_core_ids.try_resize(other.logical_cores_count_)) return false; 
        if (!scratch_domains.try_resize(other.logical_cores_count_)) return false;  
        core_id_t const *const other_cores = other.domain_core_ids_.data();
        if (other.logical_cores_count_ > 0)
            std::memcpy(scratch_core_ids.data(), other_cores, other.logical_cores_count_ * sizeof(core_id_t));
        for (std::size_t i = 0; i < other.memory_domains_count_; ++i) {
            scratch_nodes[i] = other.memory_domains_[i];
            std::ptrdiff_t const offset = other.memory_domains_[i].first_core_id - other_cores;
            scratch_nodes[i].first_core_id = scratch_core_ids.data() + offset;
        }
        for (std::size_t i = 0; i < other.compute_domains_count_; ++i) {
            scratch_domains[i] = other.compute_domains_[i];
            std::ptrdiff_t const offset = other.compute_domains_[i].first_core_id - other_cores;
            scratch_domains[i].first_core_id = scratch_core_ids.data() + offset;
        }
        memory_domains_ = std::move(scratch_nodes); 
        domain_core_ids_ = std::move(scratch_core_ids);
        compute_domains_ = std::move(scratch_domains);
        memory_domains_count_ = other.memory_domains_count_;
        logical_cores_count_ = other.logical_cores_count_;
        compute_domains_count_ = other.compute_domains_count_;
        compute_levels_count_ = other.compute_levels_count_;
        return true;
    }
    /** @brief Number of memory domains, one per NUMA node. @sa `compute_domains_count`. */
    std::size_t memory_domains_count() const noexcept { return memory_domains_count_; }
    std::size_t logical_cores_count() const noexcept { return logical_cores_count_; }
    /** @brief The memory domain at @p memory_domain_index, in [0, `memory_domains_count()`). */
    memory_domain_t const &memory_domain_at(memory_domain_index_t const memory_domain_index) const noexcept {
        assert(memory_domain_index < memory_domains_count_ && "Memory domain index is out of bounds");
        return memory_domains_[memory_domain_index];
    }
    /** @brief Number of compute domains (one per same-QoS core run within a node). */
    std::size_t compute_domains_count() const noexcept { return compute_domains_count_; }
    /** @brief Number of distinct QoS classes across all compute domains (>= 1). */
    std::size_t compute_levels_count() const noexcept { return compute_levels_count_; }
    /** @brief The compute domain at @p compute_domain_index, in [0, `compute_domains_count()`). */
    compute_domain_t const &compute_domain_at(compute_domain_index_t const compute_domain_index) const noexcept {
        assert(compute_domain_index < compute_domains_count_ && "Compute domain ID is out of bounds");
        return compute_domains_[compute_domain_index];
    }
    /** @brief The memory domain nearest a compute domain (its NUMA node); 0 if out of range. */
    memory_domain_index_t local_memory_of(compute_domain_index_t const compute_domain_index) const noexcept {
        if (compute_domain_index >= compute_domains_count_) return memory_domain_index_t {};
        return compute_domains_[compute_domain_index].memory_domain_index;
    }
    /**
     *  @brief Fills a single memory domain and compute domain covering every allowed core.
     *  @retval false only if the core count is zero or an allocation fails.
     *
     *  The uniform view used when no richer topology source exists - a build without `FU_WITH_TOPOLOGY`,
     *  or a machine the kernel reports no NUMA for. Every query then returns a sensible whole-machine
     *  answer and a `distributed_pool` degenerates to one domain, so `fu_topology_t` is usable anywhere.
     */
    bool try_harvest_portable() noexcept {
        reset();
        core_mask_t allowed;
        bool const allowed_known = try_capture_thread_cores(allowed) && allowed.count() != 0;
        std::size_t const cores = allowed_known ? allowed.count() : possible_cores();
        if (cores == 0) return false;
        dynamic_array<memory_domain_t, memory_domains_allocator_t> nodes {memory_domains_allocator_t {allocator_}};
        dynamic_array<core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        if (!nodes.try_resize(1) || !core_ids.try_resize(cores) || !domains.try_resize(1)) return false;
        core_id_t *const core_ids_ptr = core_ids.data();
        if (allowed_known) {
            std::size_t written = 0;
            std::size_t const id_space = allowed.capacity();
            for (std::size_t bit = 0; bit < id_space && written < cores; ++bit)
                if (allowed.contains(static_cast<core_id_t>(bit)))
                    core_ids_ptr[written++] = static_cast<core_id_t>(bit);
        }
        else {
            for (std::size_t i = 0; i < cores; ++i) core_ids_ptr[i] = static_cast<core_id_t>(i);
        }
        memory_domain_t &node = nodes.data()[0];
        node.memory_domain_id = 0;
        node.socket_id = 0;
        node.volume_ram = volume_ram();
        node.first_core_id = core_ids_ptr;
        node.logical_cores_count = cores;
        compute_domain_t &domain = domains.data()[0];
        domain.memory_domain_id = 0;
        domain.memory_domain_index = static_cast<memory_domain_index_t>(0);
        domain.compute_level = 0;
        domain.capacity = 0;
        domain.cache_bytes = cache_bytes_of_core(core_ids_ptr[0], core_ids_ptr, cores);
        domain.first_core_id = core_ids_ptr;
        domain.logical_cores_count = cores;
        memory_domains_ = std::move(nodes);
        domain_core_ids_ = std::move(core_ids);
        compute_domains_ = std::move(domains);
        memory_domains_count_ = 1;
        logical_cores_count_ = cores;
        compute_domains_count_ = 1;
        compute_levels_count_ = 1;
        return true;
    }
#if FU_ON_FREEBSD
    /**
     *  @brief Harvests memory domains and their cores through the in-kernel `cpuset`/NUMA framework.
     *  @sa `try_harvest` dispatches here; `try_harvest_portable` is the fallback when NUMA is absent.
     *
     *  FreeBSD ships no `libnuma`. `sysctl vm.ndomains` counts the NUMA memory domains, and
     *  `cpuset_getaffinity(CPU_WHICH_DOMAIN)` reports the cores each one owns. Every domain's cores are
     *  intersected with the set this process may actually run on, exactly as the Linux harvest does, so
     *  a jailed or `cpuset`-narrowed process never sizes a pool from cores it will never be scheduled
     *  on. Cores are not ranked by class here - FreeBSD publishes no per-core capacity - so each memory
     *  domain yields exactly one compute domain.
     */
    bool try_harvest_freebsd() noexcept {
        reset();
        core_mask_t allowed;
        bool const allowed_known = try_capture_thread_cores(allowed) && allowed.count() != 0;
        int domain_count = 0;
        std::size_t domain_count_size = sizeof(domain_count);
        if (::sysctlbyname("vm.ndomains", &domain_count, &domain_count_size, nullptr, 0) != 0 || domain_count < 1)
            return try_harvest_portable(); 
        core_mask_t domain_mask;
        if (!domain_mask.try_resize()) return false;
        std::size_t fetched_memory_domains = 0, fetched_cores = 0;
        for (int domain_id = 0; domain_id < domain_count; ++domain_id) {
            domain_mask.clear();
            if (::cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_DOMAIN, static_cast<id_t>(domain_id),
                                     domain_mask.bytes(), static_cast<cpuset_t *>(domain_mask.data())) != 0)
                continue; 
            std::size_t node_cores = 0;
            std::size_t const id_space = domain_mask.capacity();
            for (std::size_t bit = 0; bit < id_space; ++bit) {
                core_id_t const core = static_cast<core_id_t>(bit);
                if (!domain_mask.contains(core)) continue;
                if (allowed_known && !allowed.contains(core)) continue;
                ++node_cores;
            }
            if (node_cores == 0) continue;
            fetched_memory_domains += 1;
            fetched_cores += node_cores;
        }
        if (fetched_memory_domains == 0) return try_harvest_portable(); 
        dynamic_array<memory_domain_t, memory_domains_allocator_t> nodes {memory_domains_allocator_t {allocator_}};
        dynamic_array<core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        if (!nodes.try_resize(fetched_memory_domains)) return false;
        if (!core_ids.try_resize(fetched_cores)) return false;
        if (!domains.try_resize(fetched_memory_domains)) return false;
        memory_domain_t *const nodes_ptr = nodes.data();
        core_id_t *const core_ids_ptr = core_ids.data();
        compute_domain_t *const domains_ptr = domains.data();
        std::size_t const ram_per_domain = volume_ram() / fetched_memory_domains;
        std::size_t core_index = 0, node_index = 0;
        for (int domain_id = 0; domain_id < domain_count; ++domain_id) {
            domain_mask.clear();
            if (::cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_DOMAIN, static_cast<id_t>(domain_id),
                                     domain_mask.bytes(), static_cast<cpuset_t *>(domain_mask.data())) != 0)
                continue;
            std::size_t const core_begin = core_index;
            std::size_t const id_space = domain_mask.capacity();
            for (std::size_t bit = 0; bit < id_space; ++bit) {
                core_id_t const core = static_cast<core_id_t>(bit);
                if (!domain_mask.contains(core)) continue;
                if (allowed_known && !allowed.contains(core)) continue;
                core_ids_ptr[core_index++] = core;
            }
            std::size_t const node_cores = core_index - core_begin;
            if (node_cores == 0) continue;
            memory_domain_t &node = nodes_ptr[node_index];
            node.memory_domain_id = static_cast<memory_domain_id_t>(domain_id);
            node.socket_id = -1;
            node.volume_ram = ram_per_domain;
            node.first_core_id = core_ids_ptr + core_begin;
            node.logical_cores_count = node_cores;
            node.page_sizes.try_harvest(static_cast<memory_domain_id_t>(domain_id)); 
            compute_domain_t &domain = domains_ptr[node_index];
            domain.memory_domain_id = static_cast<memory_domain_id_t>(domain_id);
            domain.memory_domain_index = static_cast<memory_domain_index_t>(node_index);
            domain.compute_level = 0;
            domain.capacity = 0;
            domain.cache_bytes = cache_bytes_of_core(core_ids_ptr[core_begin], core_ids_ptr + core_begin, node_cores);
            domain.first_core_id = core_ids_ptr + core_begin;
            domain.logical_cores_count = node_cores;
            ++node_index;
        }
        memory_domains_ = std::move(nodes);
        domain_core_ids_ = std::move(core_ids);
        compute_domains_ = std::move(domains);
        memory_domains_count_ = fetched_memory_domains;
        logical_cores_count_ = fetched_cores;
        compute_domains_count_ = fetched_memory_domains;
        compute_levels_count_ = 1;
        return true;
    }
#endif 
    /**
     *  @brief Harvests CPU-memory topology - Linux NUMA nodes, or Apple Silicon performance levels.
     *  @retval false if the platform lacks topology support or the harvest failed.
     *  @retval true if the harvest was successful and the topology is ready to use.
     *
     *  Falls back to `try_harvest_portable` whenever no richer source is available, so a spawned pool
     *  always sees at least one compute and one memory domain.
     */
    bool try_harvest() noexcept {
#if FU_WITH_TOPOLOGY && FU_ON_LINUX
        reset();
        core_mask_t allowed;
        bool const allowed_known = try_capture_thread_cores(allowed) && allowed.count() != 0;
        if (!linux_has_memory_domains()) return try_harvest_portable(); 
        core_mask_t domain_mask;
        if (!domain_mask.try_resize()) return false; 
        std::size_t fetched_memory_domains = 0, fetched_cores = 0;
        memory_domain_id_t const max_numa_node_id = max_memory_domain_id();
        for (memory_domain_id_t memory_domain_id = 0; memory_domain_id <= max_numa_node_id; ++memory_domain_id) {
            std::size_t node_ram = 0;
            if (!try_read_ram_bytes_of_memory_domain(memory_domain_id, node_ram)) continue; 
            if (!try_capture_memory_domain_cores(memory_domain_id, domain_mask)) continue;  
            std::size_t node_cores = 0;
            std::size_t const id_space = domain_mask.capacity();
            for (std::size_t bit = 0; bit < id_space; ++bit) {
                core_id_t const core = static_cast<core_id_t>(bit);
                if (!domain_mask.contains(core)) continue;
                if (allowed_known && !allowed.contains(core)) continue;
                ++node_cores;
            }
            fetched_memory_domains += 1;
            fetched_cores += node_cores;
        }
        if (fetched_memory_domains == 0) return false; 
        dynamic_array<memory_domain_t, memory_domains_allocator_t> nodes {memory_domains_allocator_t {allocator_}};
        dynamic_array<core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        if (!nodes.try_resize(fetched_memory_domains)) return false;
        if (!core_ids.try_resize(fetched_cores)) return false;
        if (!domains.try_resize(fetched_cores)) return false;
        memory_domain_t *const domains_outb = nodes.data();
        core_id_t *const core_ids_ptr = core_ids.data();
        compute_domain_t *const domains_ptr = domains.data();
        std::size_t const configured_cores = possible_cores();
        if (configured_cores == 0) return false; 
        dynamic_array<std::size_t, capacities_allocator_t> capacities {capacities_allocator_t {allocator_}};
        if (!capacities.try_resize(configured_cores)) return false;
        std::size_t *const core_capacities = capacities.data();
        for (memory_domain_id_t memory_domain_id = 0, core_index = 0, node_index = 0;
             memory_domain_id <= max_numa_node_id; ++memory_domain_id) {
            std::size_t node_ram = 0;
            if (!try_read_ram_bytes_of_memory_domain(memory_domain_id, node_ram)) continue;
            if (!try_capture_memory_domain_cores(memory_domain_id, domain_mask)) continue;
            memory_domain_t &node = domains_outb[node_index];
            node.memory_domain_id = memory_domain_id;
            node.volume_ram = node_ram;
            node.first_core_id = core_ids_ptr + core_index;
            std::size_t const core_index_before = core_index;
            std::size_t const id_space = domain_mask.capacity();
            for (std::size_t bit_offset = 0; bit_offset < id_space; ++bit_offset) {
                core_id_t const core = static_cast<core_id_t>(bit_offset);
                if (!domain_mask.contains(core)) continue;
                if (allowed_known && !allowed.contains(core)) continue;
                core_ids_ptr[core_index++] = core;
            }
            node.logical_cores_count = core_index - core_index_before;
            node.socket_id = node.logical_cores_count > 0 ? socket_id_of_core(node.first_core_id[0]) : -1;
            node.page_sizes.try_harvest(memory_domain_id); 
            node_index++;
        }
        memory_domains_ = std::move(nodes);
        domain_core_ids_ = std::move(core_ids);
        memory_domains_count_ = fetched_memory_domains;
        logical_cores_count_ = fetched_cores;
        bubble_sort(memory_domains_.data(), memory_domains_count_,
                    [](memory_domain_t const &a, memory_domain_t const &b) noexcept {
                        if (a.socket_id != b.socket_id) return a.socket_id < b.socket_id;
                        if (a.logical_cores_count != b.logical_cores_count)
                            return a.logical_cores_count > b.logical_cores_count; 
                        core_id_t const a_first = a.logical_cores_count ? a.first_core_id[0] : -1;
                        core_id_t const b_first = b.logical_cores_count ? b.first_core_id[0] : -1;
                        return a_first < b_first; 
                    });
        for (std::size_t core_index = 0; core_index < logical_cores_count_; ++core_index) {
            core_id_t const core_id = domain_core_ids_[core_index];
            if (static_cast<std::size_t>(core_id) < configured_cores)
                core_capacities[static_cast<std::size_t>(core_id)] = capacity_of_core(core_id);
        }
        {
            std::size_t domains_written = 0;
            for (std::size_t node_index = 0; node_index < memory_domains_count_; ++node_index) {
                memory_domain_t &node = memory_domains_[node_index];
                core_id_t *node_cores = const_cast<core_id_t *>(node.first_core_id);
                bubble_sort(node_cores, node.logical_cores_count,
                            [core_capacities](core_id_t const &a, core_id_t const &b) noexcept {
                                return core_capacities[static_cast<std::size_t>(a)] <
                                       core_capacities[static_cast<std::size_t>(b)];
                            });
                std::size_t run_begin = 0;
                for (std::size_t core = 1; core <= node.logical_cores_count; ++core) {
                    bool const at_end = core == node.logical_cores_count;
                    bool const capacity_changed =
                        !at_end && core_capacities[static_cast<std::size_t>(node_cores[core])] !=
                                       core_capacities[static_cast<std::size_t>(node_cores[run_begin])];
                    if (!at_end && !capacity_changed) continue;
                    compute_domain_t &domain = domains_ptr[domains_written++];
                    domain.memory_domain_id = node.memory_domain_id;
                    domain.memory_domain_index = static_cast<memory_domain_index_t>(node_index);
                    domain.compute_level = core_capacities[static_cast<std::size_t>(
                        node_cores[run_begin])]; 
                    domain.capacity = core_capacities[static_cast<std::size_t>(node_cores[run_begin])];
                    domain.cache_bytes =
                        cache_bytes_of_core(node_cores[run_begin], node_cores + run_begin, core - run_begin);
                    domain.first_core_id = node_cores + run_begin;
                    domain.logical_cores_count = core - run_begin;
                    run_begin = core;
                }
            }
            compute_domains_ = std::move(domains);
            compute_domains_count_ = domains_written;
            compute_levels_count_ = dense_rank(
                domains_written,
                [core_capacities, domains_ptr](std::size_t index) noexcept {
                    return core_capacities[static_cast<std::size_t>(domains_ptr[index].first_core_id[0])];
                },
                [domains_ptr](std::size_t index, std::size_t rank) noexcept {
                    domains_ptr[index].compute_level = rank;
                });
        }
        return true; 
#endif               
#if FU_ON_APPLE
        return try_harvest_apple();
#elif FU_ON_WINDOWS
        return try_harvest_windows();
#elif FU_ON_FREEBSD
        return try_harvest_freebsd();
#else
        return try_harvest_portable();
#endif
    }
#if FU_ON_APPLE
    /**
     *  @brief Harvests the Apple Silicon topology from `sysctl` performance levels.
     *  @retval false if the machine reports no logical CPUs or an allocation failed.
     *
     *  Apple Silicon is one UMA memory domain shared by every core, so we build a single memory
     *  domain. The compute axis is cut twice: first by `hw.perflevelN`, then by `cpusperl2` within
     *  each level, because a performance level may span several L2 clusters that share no cache.
     *  A compute domain is cores sharing a QoS class @b and locality, so the cluster is the unit.
     *  Domains from one level all carry that level's `compute_level`; `hw.perflevel0` ranks highest.
     *
     *  Performance levels rank cores without rating them, and the parts they distinguish need not
     *  differ in clock - some pair equal scalar throughput with unequal cache. So `capacity` stays
     *  0 and only `cache_bytes` is populated; weighting work by the level ordinal
     *  would hand a wide-cache cluster more tasks than it can necessarily retire any faster.
     *
     *  @note macOS exposes no hard pinning - `thread_policy_set(THREAD_AFFINITY_POLICY)` returns
     *      `KERN_NOT_SUPPORTED` on arm64, and QoS classes are the only placement lever. These
     *      domains are therefore descriptive: `spawn_on` reports them, the kernel still migrates.
     */
    bool try_harvest_apple() noexcept {
        std::size_t const total_cores = apple_sysctl_uint("hw.logicalcpu");
        if (total_cores == 0) return false;
        std::size_t const memory_size = apple_sysctl_uint("hw.memsize");
        std::size_t const levels = apple_sysctl_uint("hw.nperflevels");
        std::size_t nonempty_levels = 0;
        for (std::size_t level = 0; level < levels; ++level) {
            char name[64];
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.logicalcpu", level);
            if (apple_sysctl_uint(name)) nonempty_levels += 1;
        }
        if (nonempty_levels == 0) nonempty_levels = 1; 
        dynamic_array<memory_domain_t, memory_domains_allocator_t> nodes {memory_domains_allocator_t {allocator_}};
        dynamic_array<core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        if (!nodes.try_resize(1)) return false;
        if (!core_ids.try_resize(total_cores)) return false;
        if (!domains.try_resize(total_cores)) return false;
        memory_domain_t *const domains_outb = nodes.data();
        core_id_t *const core_ids_ptr = core_ids.data();
        compute_domain_t *const domains_ptr = domains.data();
        for (std::size_t i = 0; i < total_cores; ++i) core_ids_ptr[i] = static_cast<core_id_t>(i);
        memory_domain_t &node = domains_outb[0];
        node.memory_domain_id = 0;
        node.socket_id = 0;
        node.volume_ram = memory_size;
        node.first_core_id = core_ids_ptr;
        node.logical_cores_count = total_cores;
        std::size_t core_offset = 0, domains_written = 0, levels_written = 0;
        for (std::size_t level = 0; level < levels && core_offset < total_cores; ++level) {
            char name[64];
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.logicalcpu", level);
            std::size_t level_cores = apple_sysctl_uint(name);
            if (level_cores == 0) continue;
            if (core_offset + level_cores > total_cores) level_cores = total_cores - core_offset;
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.l2cachesize", level);
            std::size_t const level_cache_bytes = apple_sysctl_uint(name);
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.cpusperl2", level);
            std::size_t cores_per_cluster = apple_sysctl_uint(name);
            if (cores_per_cluster == 0 || cores_per_cluster > level_cores) cores_per_cluster = level_cores;
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
            std::snprintf(name, sizeof(name), "hw.perflevel%zu.name", level);
            char level_name[32];
            apple_sysctl_string(name, level_name, sizeof(level_name));
            core_quality_t const level_quality = apple_core_quality_from_name(level_name);
#endif
            std::size_t const level_rank = nonempty_levels - 1 - levels_written;
            for (std::size_t cut = 0; cut < level_cores; cut += cores_per_cluster) {
                std::size_t const cluster_cores = (level_cores - cut) < cores_per_cluster 
                                                      ? (level_cores - cut)
                                                      : cores_per_cluster;
                compute_domain_t &domain = domains_ptr[domains_written];
                domain.memory_domain_id = 0;
                domain.memory_domain_index = static_cast<memory_domain_index_t>(0);
                domain.compute_level = level_rank; 
#if FU_WITH_PLACE_THREADS_BY_CORE_CLASS
                domain.apple_core_quality = level_quality;
#endif
                domain.capacity = 0;
                domain.cache_bytes = level_cache_bytes; 
                domain.first_core_id = core_ids_ptr + core_offset + cut;
                domain.logical_cores_count = cluster_cores;
                domains_written += 1;
            }
            core_offset += level_cores;
            levels_written += 1;
        }
        if (domains_written == 0) {
            compute_domain_t &domain = domains_ptr[0];
            domain.memory_domain_id = 0;
            domain.memory_domain_index = static_cast<memory_domain_index_t>(0);
            domain.compute_level = 0;
            domain.capacity = 0;
            domain.cache_bytes = cache_bytes_of_core(core_ids_ptr[0], core_ids_ptr, total_cores);
            domain.first_core_id = core_ids_ptr;
            domain.logical_cores_count = total_cores;
            domains_written = 1;
            levels_written = 1;
        }
        memory_domains_ = std::move(nodes);
        domain_core_ids_ = std::move(core_ids);
        compute_domains_ = std::move(domains);
        memory_domains_count_ = 1;
        logical_cores_count_ = total_cores;
        compute_domains_count_ = domains_written;
        compute_levels_count_ = levels_written; 
        return true;
    }
#endif 
#if FU_ON_WINDOWS
    /**
     *  @brief Harvests the Windows topology from `GetLogicalProcessorInformationEx`.
     *  @retval false if the machine reports no NUMA node or an allocation failed.
     *
     *  Windows describes a machine in the same two axes this library already uses. A @b processor
     *  @b group holds at most 64 logical processors sharing one `KAFFINITY` mask; groups are cut along
     *  NUMA boundaries, so a NUMA node maps to one memory domain. Within a node the compute axis is cut
     *  by @b efficiency @b class - the kernel's rank of a core's performance, where a higher class is
     *  more performant - so a hybrid P/E chip yields one compute domain per class, exactly as Apple's
     *  performance levels do. Non-hybrid chips report class 0 for every core and collapse to a single
     *  compute domain per node.
     *
     *  Rather than a per-processor scratch table, the harvest accumulates one mask per (processor group,
     *  efficiency class) - a @ref `win_group_class_cell_t` - so its working set is tiny and it reads the
     *  compute domains straight out of masks, the unit Windows itself speaks in.
     *
     *  Efficiency class is an ordinal, not a magnitude - it ranks cores without rating them - so
     *  `capacity` stays 0 and callers weigh domains by `logical_cores_count`, mirroring the
     *  Apple path. `cache_bytes` is the largest private (L1/L2) cache the kernel reports for the class.
     *
     *  @note A `core_id_t` here is not a flat index: it packs the (group, in-group bit) pair via
     *      `win_encode_core_id`, which `try_pin_thread_to_cores` decodes back into a `GROUP_AFFINITY`.
     *  @note A NUMA node spanning several processor groups (the largest servers) is enumerated in full
     *      where the SDK exposes `GroupMasks[]`; @sa `win_numa_for_each_group`.
     */
    bool try_harvest_windows() noexcept {
        auto query = [](LOGICAL_PROCESSOR_RELATIONSHIP relationship, DWORD &out_len) -> BYTE * {
            DWORD len = 0;
            ::GetLogicalProcessorInformationEx(relationship, nullptr, &len);
            if (len == 0) return nullptr; 
            BYTE *buffer = static_cast<BYTE *>(std::malloc(len));
            if (!buffer) return nullptr; 
            auto *typed = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer);
            if (!::GetLogicalProcessorInformationEx(relationship, typed, &len)) {
                std::free(buffer);
                return nullptr; 
            }
            out_len = len;
            return buffer;
        };
        using cell_allocator_t =
            typename std::allocator_traits<allocator_t>::template rebind_alloc<win_group_class_cell_t>;
        cell_allocator_t cell_alloc {allocator_};
        BYTE *numa_buf = nullptr, *package_buf = nullptr;
        win_group_class_cell_t *cells = nullptr; 
        dynamic_array<memory_domain_t, memory_domains_allocator_t> nodes {memory_domains_allocator_t {allocator_}};
        dynamic_array<core_id_t, cores_allocator_t> core_ids {cores_allocator_t {allocator_}};
        dynamic_array<compute_domain_t, domains_allocator_t> domains {domains_allocator_t {allocator_}};
        memory_domain_t *domains_outb = nullptr;
        core_id_t *core_ids_ptr = nullptr;
        compute_domain_t *domains_ptr = nullptr;
        DWORD numa_len = 0, cores_len = 0, cache_len = 0, package_len = 0;
        std::size_t class_count = 1, cell_count = 0; 
        std::size_t counted_nodes = 0, counted_cores = 0;
        std::size_t core_cursor = 0, node_index = 0, domain_cursor = 0, levels = 1;
        WORD const group_span = ::GetActiveProcessorGroupCount() ? ::GetActiveProcessorGroupCount() : 1;
        {
            BYTE *cores_buf = query(RelationProcessorCore, cores_len);
            if (!cores_buf) goto failed_harvest; 
            for (DWORD offset = 0; offset < cores_len;) {
                auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(cores_buf + offset);
                if (record->Relationship == RelationProcessorCore &&
                    static_cast<std::size_t>(record->Processor.EfficiencyClass) + 1 > class_count)
                    class_count = static_cast<std::size_t>(record->Processor.EfficiencyClass) + 1;
                if (record->Size == 0) break; 
                offset += record->Size;
            }
            cell_count = static_cast<std::size_t>(group_span) * class_count;
            cells = cell_alloc.allocate(cell_count);
            if (!cells) {
                std::free(cores_buf);
                goto failed_harvest; 
            }
            for (std::size_t i = 0; i < cell_count; ++i) cells[i] = win_group_class_cell_t {};
            for (DWORD offset = 0; offset < cores_len;) {
                auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(cores_buf + offset);
                if (record->Relationship == RelationProcessorCore) {
                    std::size_t const cls = record->Processor.EfficiencyClass;
                    for (WORD g = 0; g < record->Processor.GroupCount; ++g) {
                        GROUP_AFFINITY const affinity = record->Processor.GroupMask[g];
                        if (affinity.Group < group_span)
                            cells[static_cast<std::size_t>(affinity.Group) * class_count + cls].mask |= affinity.Mask;
                    }
                }
                if (record->Size == 0) break;
                offset += record->Size;
            }
            std::free(cores_buf);
        }
        if (BYTE *cache_buf = query(RelationCache, cache_len)) {
            for (DWORD offset = 0; offset < cache_len;) {
                auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(cache_buf + offset);
                if (record->Relationship == RelationCache && record->Cache.Level <= 2) {
                    GROUP_AFFINITY const affinity = record->Cache.GroupMask;
                    std::size_t const size_bytes = static_cast<std::size_t>(record->Cache.CacheSize);
                    if (affinity.Group < group_span)
                        for (std::size_t c = 0; c < class_count; ++c) {
                            win_group_class_cell_t &cell =
                                cells[static_cast<std::size_t>(affinity.Group) * class_count + c];
                            if ((affinity.Mask & cell.mask) && size_bytes > cell.cache_bytes)
                                cell.cache_bytes = size_bytes;
                        }
                }
                if (record->Size == 0) break;
                offset += record->Size;
            }
            std::free(cache_buf);
        }
        package_buf = query(RelationProcessorPackage, package_len);
        numa_buf = query(RelationNumaNode, numa_len);
        if (!numa_buf) goto failed_harvest; 
        for (DWORD offset = 0; offset < numa_len;) {
            auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(numa_buf + offset);
            if (record->Relationship == RelationNumaNode) {
                counted_nodes += 1;
                win_numa_for_each_group(record->NumaNode, [&](WORD group, KAFFINITY node_mask) noexcept {
                    if (group >= group_span) return; 
                    for (std::size_t c = 0; c < class_count; ++c)
                        counted_cores += popcount(static_cast<KAFFINITY>(
                            node_mask & cells[static_cast<std::size_t>(group) * class_count + c].mask));
                });
            }
            if (record->Size == 0) break;
            offset += record->Size;
        }
        if (counted_nodes == 0 || counted_cores == 0) goto failed_harvest; 
        if (!nodes.try_resize(counted_nodes)) goto failed_harvest;    
        if (!core_ids.try_resize(counted_cores)) goto failed_harvest; 
        if (!domains.try_resize(counted_cores)) goto failed_harvest;  
        domains_outb = nodes.data();
        core_ids_ptr = core_ids.data();
        domains_ptr = domains.data();
        for (DWORD offset = 0; offset < numa_len;) {
            auto *record = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(numa_buf + offset);
            if (record->Relationship != RelationNumaNode) {
                if (record->Size == 0) break;
                offset += record->Size;
                continue;
            }
            NUMA_NODE_RELATIONSHIP const &memory_domain = record->NumaNode;
            std::size_t const node_first_core = core_cursor;
            for (std::size_t class_step = 0; class_step < class_count; ++class_step) {
                std::size_t const cls = class_count - 1 - class_step; 
                std::size_t const domain_first_core = core_cursor;
                std::size_t domain_cache_bytes = 0;
                win_numa_for_each_group(memory_domain, [&](WORD group, KAFFINITY node_mask) noexcept {
                    if (group >= group_span) return; 
                    win_group_class_cell_t const &cell = cells[static_cast<std::size_t>(group) * class_count + cls];
                    KAFFINITY const domain_mask = node_mask & cell.mask;
                    if (!domain_mask) return;
                    if (cell.cache_bytes > domain_cache_bytes) domain_cache_bytes = cell.cache_bytes;
                    for (unsigned bit = 0; bit < win_processors_per_group_k; ++bit)
                        if (domain_mask & (static_cast<KAFFINITY>(1) << bit))
                            core_ids_ptr[core_cursor++] = win_encode_core_id(group, bit);
                });
                if (core_cursor == domain_first_core) continue; 
                compute_domain_t &domain = domains_ptr[domain_cursor++];
                domain.memory_domain_id = static_cast<memory_domain_id_t>(memory_domain.NodeNumber);
                domain.memory_domain_index = static_cast<memory_domain_index_t>(node_index);
                domain.compute_level = cls; 
                domain.capacity = 0;
                domain.cache_bytes = domain_cache_bytes;
                domain.first_core_id = core_ids_ptr + domain_first_core;
                domain.logical_cores_count = core_cursor - domain_first_core;
            }
            ULONGLONG available_bytes = 0;
            (void)::GetNumaAvailableMemoryNodeEx(static_cast<USHORT>(memory_domain.NodeNumber), &available_bytes);
            memory_domain_t &node = domains_outb[node_index];
            node.memory_domain_id = static_cast<memory_domain_id_t>(memory_domain.NodeNumber);
            node.socket_id = win_socket_for_node(package_buf, package_len, memory_domain,
                                                 static_cast<socket_id_t>(memory_domain.NodeNumber));
            node.volume_ram =
                static_cast<std::size_t>(available_bytes); 
            node.first_core_id = core_ids_ptr + node_first_core;
            node.logical_cores_count = core_cursor - node_first_core;
            node.page_sizes.try_harvest(node.memory_domain_id); 
            node_index += 1;
            if (record->Size == 0) break;
            offset += record->Size;
        }
        std::free(package_buf);
        package_buf = nullptr;
        std::free(numa_buf);
        numa_buf = nullptr; 
        cell_alloc.deallocate(cells, cell_count);
        cells = nullptr;
        if (domain_cursor != 0)
            levels = dense_rank(
                domain_cursor,
                [domains_ptr](std::size_t i) noexcept {
                    return static_cast<std::size_t>(domains_ptr[i].compute_level);
                },
                [domains_ptr](std::size_t i, std::size_t rank) noexcept { domains_ptr[i].compute_level = rank; });
        memory_domains_ = std::move(nodes);
        domain_core_ids_ = std::move(core_ids);
        compute_domains_ = std::move(domains);
        memory_domains_count_ = counted_nodes;
        logical_cores_count_ = core_cursor;
        compute_domains_count_ = domain_cursor;
        compute_levels_count_ = levels;
        return true;
    failed_harvest: 
        if (cells) cell_alloc.deallocate(cells, cell_count);
        std::free(numa_buf);
        std::free(package_buf);
        return false;
    }
#endif 
};
using machine_topology_t = machine_topology<>;
} 
} 
