module SysInfo

using ..Core: register_benchmark, available_cpu_threads
using CUDA
using Printf: @sprintf

function run_sysinfo(args)
    @info "Collecting System Information..."

    results = Dict{String, Any}()

    # CPU and system info (stdlib only)
    cpus = Sys.cpu_info()
    if !isempty(cpus)
        results["cpu_model"] = cpus[1].model
    end
    results["cpu_arch"] = string(Sys.CPU_NAME)
    results["cpu_threads"] = available_cpu_threads()
    results["ram_total"] = Base.format_bytes(Int(Sys.total_memory()))
    results["ram_free"] = Base.format_bytes(Int(Sys.free_memory()))

    # GPU info
    if CUDA.functional()
        dev = CUDA.device()
        results["gpu_name"] = CUDA.name(dev)
        results["compute_capability"] = string(CUDA.capability(dev))
        results["vram_total"] = Base.format_bytes(Int(CUDA.totalmem(dev)))
        results["vram_free"] = Base.format_bytes(Int(CUDA.available_memory()))
        results["gpu_uuid"] = string(CUDA.uuid(dev))
        results["gpu_count"] = length(CUDA.devices())
        results["cuda_driver"] = string(CUDA.driver_version())

        # PCI bus ID
        domain = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
        bus    = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
        devid  = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
        results["pci_bus_id"] = @sprintf("%04x:%02x:%02x.0", domain, bus, devid)

        # All GPUs
        gpu_list = Dict{String, Any}[]
        for d in CUDA.devices()
            dom = CUDA.attribute(d, CUDA.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
            b   = CUDA.attribute(d, CUDA.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
            did = CUDA.attribute(d, CUDA.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
            push!(gpu_list, Dict(
                "id" => CUDA.deviceid(d),
                "name" => CUDA.name(d),
                "uuid" => string(CUDA.uuid(d)),
                "pci_bus_id" => @sprintf("%04x:%02x:%02x.0", dom, b, did),
                "compute_capability" => string(CUDA.capability(d)),
                "vram_total" => Base.format_bytes(Int(CUDA.totalmem(d)))
            ))
        end
        results["gpus"] = gpu_list
    end

    return results
end

register_benchmark(
    "sysinfo",
    "Display system and GPU hardware information without running benchmarks",
    run_sysinfo
)

end # module
