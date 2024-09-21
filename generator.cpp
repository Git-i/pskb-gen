#include "generator.h"
#include "ktx.h"
#include "vkformat.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdexcept>
#include <memory>
#include <ranges>
#include <set>

#include "float16.h"
#include "rhi_sc.h"
#include "ShaderReflect.h"

namespace RHI_SC = RHI::ShaderCompiler;
std::unique_ptr<std::uint16_t[]> image_4ch_half(const float* const f, uint32_t width, uint32_t height)
{
    size_t n_floats = width * height * 4;
    std::unique_ptr<std::uint16_t[]> ptr = std::make_unique<std::uint16_t[]>(n_floats);
    for(size_t i : std::views::iota(size_t{0}, n_floats))
    {
        ptr[i] = half::float_to_half(std::bit_cast<uint32_t>(f[i]));
    }
    return ptr;
}
static RHI::creation_result<RHI::Texture> CreateCubeMap(RHI::Weak<RHI::Device> device, uint32_t width, uint32_t height, RHI::TextureUsage ex_use, uint32_t num_mips = 1)
{
    RHI::AutomaticAllocationInfo inf {RHI::AutomaticAllocationCPUAccessMode::Random};
    return device->CreateTexture(RHI::TextureDesc{
            .type = RHI::TextureType::Texture2D,
            .width = width,
            .height = height,
            .depthOrArraySize = 6,
            .format = RHI::Format::R16G16B16A16_FLOAT,
            .mipLevels = num_mips,
            .sampleCount = 1,
            .mode = RHI::TextureTilingMode::Linear,
            .optimizedClearValue = nullptr,
            .usage = RHI::TextureUsage::CubeMap | RHI::TextureUsage::CopyDst | ex_use
        }, nullptr, nullptr, &inf, 0, RHI::ResourceType::Automatic).value();
}
static std::vector<char> CompileShader(std::filesystem::path filename, RHI::API api)
{
    static auto opt = RHI_SC::CompileOptions::New();
    static auto cmp = RHI_SC::Compiler::New();
    std::vector<char> out;
    if(const auto [warning_count, messages, error] = cmp->CompileToBuffer(api, RHI_SC::ShaderSource{.source = filename, .stage = RHI::ShaderStage::Compute}, opt, out);
        error != RHI_SC::CompilationError::None)
        throw std::runtime_error(messages);
    return out;
}
void generator::create_base()
{
    auto& device = ctx.device;
    width = 512; height = 512;
    if(mode == input_mode::folder)
    {
        void* data[6];
        const char* file_names[] = {"px", "nx", "py", "ny", "pz", "nz"};
        int ex_w = 0;
        int ex_h = 0;
        for(uint32_t i = 0; i < 6; i++)
        {
            std::string full = input + file_names[i] + input_ext;
            int w, h ,c;
            data[i] = stbi_loadf(full.c_str(), &w, &h, &c, 4);
            if(ex_w != 0 && w != ex_w) throw std::runtime_error("Incompatible sizes");
            if(ex_h!= 0 && h != ex_h) throw std::runtime_error("Incompatible sizes");
            ex_w = w;
            ex_h = h;
            if(!data[i]) throw std::runtime_error("File not found");
        }
        width = ex_w; height = ex_h;

        base = CreateCubeMap(device, width, height, RHI::TextureUsage::SampledImage).value();
        const uint32_t face_size = (width * height * sizeof(uint16_t) * 4);
        uint32_t idx = 0;
        for(auto & i : data)
        {
            const auto mapped = static_cast<uint8_t*>(base->Map(RHI::Aspect::COLOR_BIT, 0, idx).value());
            auto ptr = image_4ch_half(static_cast<float*>(i), width, height);
            memcpy(mapped, ptr.get(), face_size);
            base->UnMap(RHI::Aspect::COLOR_BIT, 0, idx);
            ++idx;
        }
    }
    else
    {
        width = 512, height = 512;
        base = CreateCubeMap(device, width, height, RHI::TextureUsage::SampledImage | RHI::TextureUsage::StorageImage).value();
        std::vector out = CompileShader("../shaders/eq_to_cubemap.hlsl", ctx.inst->GetInstanceAPI());
        int x, y, nc;
        const void* data = stbi_loadf(input.c_str(), &x, &y, &nc, 4);
        const RHI::AutomaticAllocationInfo inf {RHI::AutomaticAllocationCPUAccessMode::Sequential};
        RHI::Ptr<RHI::Texture> tex = device->CreateTexture(RHI::TextureDesc{
            .type = RHI::TextureType::Texture2D,
            .width = static_cast<uint32_t>(x),
            .height = static_cast<uint32_t>(y),
            .depthOrArraySize = 1,
            .format = RHI::Format::R32G32B32A32_FLOAT,
            .mipLevels = 1,
            .sampleCount = 1,
            .mode = RHI::TextureTilingMode::Linear,
            .optimizedClearValue = nullptr,
            .usage = RHI::TextureUsage::SampledImage,
            .layout = RHI::ResourceLayout::PREINITIALIZED
        }, nullptr, nullptr, &inf, 0, RHI::ResourceType::Automatic).value();

        const auto eq_ptr = static_cast<uint8_t*>(tex->Map(RHI::Aspect::COLOR_BIT, 0, 0).value());
        memcpy(eq_ptr, data, x * y * sizeof(float) * 4);
        tex->UnMap(RHI::Aspect::COLOR_BIT, 0, 0);


        RHI::Ptr<RHI::DescriptorSetLayout> dsl;
        auto refl = RHI::ShaderReflection::CreateFromMemory({out.data(), out.size()}).value();
        auto [rsdesc, _1, _2] = RHI::ShaderReflection::FillRootSignatureDesc({&refl, 1}, {}, std::nullopt);
        auto rs = device->CreateRootSignature(rsdesc, &dsl).value();
        auto shader = device->CreateComputePipeline(RHI::ComputePipelineDesc{
            .CS = {{out.begin(), out.end()}},
            .mode = RHI::ShaderMode::Memory,
            .rootSig = rs
        }).value();
        const RHI::Ptr<RHI::DescriptorSet> set = device->CreateDescriptorSets(ctx.descriptorHeap, 1, &dsl).value()[0];
        auto v1 = device->CreateTextureView(
                {
                    .type = RHI::TextureViewType::Texture2D,
                    .format = RHI::Format::R32G32B32A32_FLOAT,
                    .texture = tex,
                    .range = RHI::SubResourceRange{
                        .imageAspect = RHI::Aspect::COLOR_BIT,
                        .IndexOrFirstMipLevel = 0,
                        .NumMipLevels = 1,
                        .FirstArraySlice = 0,
                        .NumArraySlices = 1
                    }
                }).value();
        auto v2 = device->CreateTextureView(
                    {
                        .type = RHI::TextureViewType::Texture2DArray,
                        .format = RHI::Format::R16G16B16A16_FLOAT,
                        .texture = base,
                        .range = RHI::SubResourceRange{
                            .imageAspect = RHI::Aspect::COLOR_BIT,
                            .IndexOrFirstMipLevel = 0,
                            .NumMipLevels = 1,
                            .FirstArraySlice = 0,
                            .NumArraySlices = 6
                        }
                    }).value();
        RHI::DescriptorTextureInfo tex_info[2]
        {
            {
                .texture = v1
            },
            {
                .texture = v2
            }
        };
        device->CreateSampler(RHI::SamplerDesc{
            .AddressU = RHI::AddressMode::Clamp,
            .AddressV = RHI::AddressMode::Clamp,
            .AddressW = RHI::AddressMode::Clamp,
            .anisotropyEnable = false,
            .compareEnable = false,
            .compareFunc = RHI::ComparisonFunc::Never,
            .minFilter = RHI::Filter::Linear,
            .magFilter = RHI::Filter::Linear,
            .mipFilter = RHI::Filter::Linear,
            .maxAnisotropy = 0,
            .minLOD = 0,
            .maxLOD = FLT_MAX,
            .mipLODBias = 0
        }, ctx.samplerHeap->GetCpuHandle());
        RHI::DescriptorSamplerInfo sampler_info{
            .heapHandle = ctx.samplerHeap->GetCpuHandle()
        };
        RHI::DescriptorSetUpdateDesc updates[3]
        {
            {
                .binding = 0,
                .arrayIndex = 0,
                .numDescriptors = 1,
                .type = RHI::DescriptorType::CSTexture,
                .textureInfos = tex_info + 1
            },
            {
                .binding = 1,
                .arrayIndex = 0,
                .numDescriptors = 1,
                .type = RHI::DescriptorType::SampledTexture,
                .textureInfos = tex_info
            },
            {
                .binding = 2,
                .arrayIndex = 0,
                .numDescriptors = 1,
                .type = RHI::DescriptorType::Sampler,
                .samplerInfos = &sampler_info
            }
        };
        device->UpdateDescriptorSet(updates, set);
        ctx.list->Begin(ctx.allocator);
        std::array<RHI::TextureMemoryBarrier, 2> img_barr = {
            RHI::TextureMemoryBarrier{
                .AccessFlagsBefore = RHI::ResourceAcessFlags::NONE,
                .AccessFlagsAfter = RHI::ResourceAcessFlags::SHADER_WRITE,
                .oldLayout = RHI::ResourceLayout::UNDEFINED,
                .newLayout = RHI::ResourceLayout::GENERAL,
                .texture = base,
                .previousQueue = RHI::QueueFamily::Ignored,
                .nextQueue = RHI::QueueFamily::Ignored,
                .subresourceRange = RHI::SubResourceRange{
                    .imageAspect = RHI::Aspect::COLOR_BIT,
                    .IndexOrFirstMipLevel = 0,
                    .NumMipLevels = 1,
                    .FirstArraySlice = 0,
                    .NumArraySlices = 6
                }
            },
            RHI::TextureMemoryBarrier{
                .AccessFlagsBefore = RHI::ResourceAcessFlags::NONE,
                .AccessFlagsAfter = RHI::ResourceAcessFlags::SHADER_READ,
                .oldLayout = RHI::ResourceLayout::PREINITIALIZED,
                .newLayout = RHI::ResourceLayout::SHADER_READ_ONLY_OPTIMAL,
                .texture = tex,
                .previousQueue = RHI::QueueFamily::Ignored,
                .nextQueue = RHI::QueueFamily::Ignored,
                .subresourceRange = RHI::SubResourceRange{
                    .imageAspect = RHI::Aspect::COLOR_BIT,
                    .IndexOrFirstMipLevel = 0,
                    .NumMipLevels = 1,
                    .FirstArraySlice = 0,
                    .NumArraySlices = 1
                }
            }
        };
        ctx.list->PipelineBarrier(RHI::PipelineStage::TOP_OF_PIPE_BIT, RHI::PipelineStage::COMPUTE_SHADER_BIT, {}, img_barr);
        ctx.list->SetRootSignature(rs);
        ctx.list->BindComputeDescriptorSet(set, 0);
        ctx.list->SetComputePipeline(shader);
        ctx.list->Dispatch(width, height, 6);
        ctx.list->End();
        ctx.queue->ExecuteCommandLists(&ctx.list->ID, 1);
        ctx.queue->SignalFence(ctx.fence, ++ctx.fence_value);
        ctx.fence->Wait(ctx.fence_value);
    }
}
void generator::create_ir(uint32_t ir_size)
{
    auto& device = ctx.device;
    ir = CreateCubeMap(device, ir_size, ir_size, RHI::TextureUsage::StorageImage).value();
    std::vector code = CompileShader("../shaders/convolute_cubemap.hlsl", ctx.inst->GetInstanceAPI());
    auto refl = RHI::ShaderReflection::CreateFromMemory({code.data(), code.size()}).value();
    auto [rsdesc, _1, _2] = RHI::ShaderReflection::FillRootSignatureDesc({&refl, 1}, {}, std::nullopt);
    RHI::Ptr<RHI::DescriptorSetLayout> layout;
    auto rs = device->CreateRootSignature(rsdesc, &layout).value();
    auto set = device->CreateDescriptorSets(ctx.descriptorHeap, 1, &layout).value()[0];
    auto v1 = device->CreateTextureView(RHI::TextureViewDesc{
        .type = RHI::TextureViewType::TextureCube,
		.format = RHI::Format::R16G16B16A16_FLOAT,
		.texture = base,
		.range {
            .imageAspect = RHI::Aspect::COLOR_BIT,
            .IndexOrFirstMipLevel = 0,
            .NumMipLevels = 1,
            .FirstArraySlice = 0,
            .NumArraySlices = 6,
        }
    }).value();
    auto v2 = device->CreateTextureView(RHI::TextureViewDesc{
        .type = RHI::TextureViewType::Texture2DArray,
		.format = RHI::Format::R16G16B16A16_FLOAT,
		.texture = ir,
		.range {
            .imageAspect = RHI::Aspect::COLOR_BIT,
            .IndexOrFirstMipLevel = 0,
            .NumMipLevels = 1,
            .FirstArraySlice = 0,
            .NumArraySlices = 6,
        }
    }).value();
    std::array tex_infos{
        RHI::DescriptorTextureInfo{
            .texture = v1
        },
        RHI::DescriptorTextureInfo{
            .texture = v2
        }
    };
    RHI::DescriptorSamplerInfo sampler_info {.heapHandle = ctx.samplerHeap->GetCpuHandle()};
    std::array writes {
        RHI::DescriptorSetUpdateDesc{
            .binding = 0,
		    .arrayIndex = 0,
		    .numDescriptors = 1,
		    .type = RHI::DescriptorType::SampledTexture,
            .textureInfos = &tex_infos[0]
        },
        RHI::DescriptorSetUpdateDesc{
            .binding = 1,
		    .arrayIndex = 0,
		    .numDescriptors = 1,
		    .type = RHI::DescriptorType::CSTexture,
            .textureInfos = &tex_infos[1]
        },
        RHI::DescriptorSetUpdateDesc{
            .binding = 2,
            .arrayIndex = 0,
            .numDescriptors = 1,
            .type = RHI::DescriptorType::Sampler,
            .samplerInfos = &sampler_info
        }
    };
    device->UpdateDescriptorSet(writes, set);
    auto shader = device->CreateComputePipeline(RHI::ComputePipelineDesc{
        .CS = {{code.data(), code.size()}},
        .mode = RHI::ShaderMode::Memory,
        .rootSig = rs
    }).value();
    ctx.list->Begin(ctx.allocator);
        std::array<RHI::TextureMemoryBarrier, 2> img_barr = {
            RHI::TextureMemoryBarrier{
                .AccessFlagsBefore = RHI::ResourceAcessFlags::SHADER_WRITE,
                .AccessFlagsAfter = RHI::ResourceAcessFlags::SHADER_READ,
                .oldLayout = RHI::ResourceLayout::GENERAL,
                .newLayout = RHI::ResourceLayout::SHADER_READ_ONLY_OPTIMAL,
                .texture = base,
                .previousQueue = RHI::QueueFamily::Ignored,
                .nextQueue = RHI::QueueFamily::Ignored,
                .subresourceRange = RHI::SubResourceRange{
                    .imageAspect = RHI::Aspect::COLOR_BIT,
                    .IndexOrFirstMipLevel = 0,
                    .NumMipLevels = 1,
                    .FirstArraySlice = 0,
                    .NumArraySlices = 6
                }
            },
            RHI::TextureMemoryBarrier{
                .AccessFlagsBefore = RHI::ResourceAcessFlags::NONE,
                .AccessFlagsAfter = RHI::ResourceAcessFlags::SHADER_WRITE,
                .oldLayout = RHI::ResourceLayout::UNDEFINED,
                .newLayout = RHI::ResourceLayout::GENERAL,
                .texture = ir,
                .previousQueue = RHI::QueueFamily::Ignored,
                .nextQueue = RHI::QueueFamily::Ignored,
                .subresourceRange = RHI::SubResourceRange{
                    .imageAspect = RHI::Aspect::COLOR_BIT,
                    .IndexOrFirstMipLevel = 0,
                    .NumMipLevels = 1,
                    .FirstArraySlice = 0,
                    .NumArraySlices = 6
                }
            }
        };
        ctx.list->PipelineBarrier(RHI::PipelineStage::COMPUTE_SHADER_BIT, RHI::PipelineStage::COMPUTE_SHADER_BIT, {}, img_barr);
        ctx.list->SetRootSignature(rs);
        ctx.list->BindComputeDescriptorSet(set, 0);
        ctx.list->SetComputePipeline(shader);
        ctx.list->Dispatch(ir_size, ir_size, 6);
        ctx.list->End();
        ctx.queue->ExecuteCommandLists(&ctx.list->ID, 1);
        ctx.queue->SignalFence(ctx.fence, ++ctx.fence_value);
        ctx.fence->Wait(ctx.fence_value );
    
}
void generator::create_pf()
{
    auto& device = ctx.device;
    pf = CreateCubeMap(device, 128, 128, RHI::TextureUsage::StorageImage, 5).value();
    std::vector code = CompileShader("../shaders/prefilter_specular.hlsl", ctx.inst->GetInstanceAPI());
    RHI::Ptr refl = RHI::ShaderReflection::CreateFromMemory({code.data(), code.size()}).value();
    auto [desc, _1, _2] = RHI::ShaderReflection::FillRootSignatureDesc({&refl, 1}, {}, 3);
    RHI::Ptr<RHI::DescriptorSetLayout> layout;
    auto rs = device->CreateRootSignature(desc, &layout).value();
    auto shader = device->CreateComputePipeline({.CS = {{code.data(), code.size()}}, .mode = RHI::ShaderMode::Memory, .rootSig = rs}).value();
    auto base_view = device->CreateTextureView(RHI::TextureViewDesc{
        .type = RHI::TextureViewType::TextureCube,
        .format = RHI::Format::R16G16B16A16_FLOAT,
        .texture = base,
        .range = RHI::SubResourceRange{
            .imageAspect = RHI::Aspect::COLOR_BIT,
            .IndexOrFirstMipLevel = 0,
            .NumMipLevels = 1,
            .FirstArraySlice = 0,
            .NumArraySlices = 6
        }
    }).value();
    auto mip0_view = device->CreateTextureView(RHI::TextureViewDesc{
        .type = RHI::TextureViewType::Texture2DArray,
            .format = RHI::Format::R16G16B16A16_FLOAT,
            .texture = pf,
            .range = RHI::SubResourceRange{
                .imageAspect = RHI::Aspect::COLOR_BIT,
                .IndexOrFirstMipLevel = 0,
                .NumMipLevels = 1,
                .FirstArraySlice = 0,
                .NumArraySlices = 6
            }
    }).value();
    auto set = device->CreateDescriptorSets(ctx.descriptorHeap, 1, &layout).value()[0];
    RHI::DescriptorTextureInfo base_texture_info{base_view};
    RHI::DescriptorTextureInfo mip0_texture_info{mip0_view};
    RHI::DescriptorSamplerInfo sampler_info {.heapHandle = ctx.samplerHeap->GetCpuHandle()};
    device->UpdateDescriptorSet({
        {
            RHI::DescriptorSetUpdateDesc{
                .binding = 0,
                .arrayIndex = 0,
                .numDescriptors = 1,
                .type = RHI::DescriptorType::SampledTexture,
                .textureInfos = &base_texture_info
            },
            RHI::DescriptorSetUpdateDesc{
                .binding = 1,
                .arrayIndex = 0,
                .numDescriptors = 1,
                .type = RHI::DescriptorType::CSTexture,
                .textureInfos = &mip0_texture_info
            },
            RHI::DescriptorSetUpdateDesc{
                .binding = 2,
                .arrayIndex = 0,
                .numDescriptors = 1,
                .type = RHI::DescriptorType::Sampler,
                .samplerInfos = &sampler_info
            },
        }
    }, set);
    ctx.list->Begin(ctx.allocator);
    ctx.list->SetRootSignature(rs);
    ctx.list->BindComputeDescriptorSet(set, 0);
    ctx.list->SetComputePipeline(shader);
    ctx.list->PushConstant(3, std::bit_cast<uint32_t>(.0f), 0);
    std::array img_barr = {
        RHI::TextureMemoryBarrier{
            .AccessFlagsBefore = RHI::ResourceAcessFlags::NONE,
            .AccessFlagsAfter = RHI::ResourceAcessFlags::SHADER_WRITE,
            .oldLayout = RHI::ResourceLayout::UNDEFINED,
            .newLayout = RHI::ResourceLayout::GENERAL,
            .texture = pf,
            .previousQueue = RHI::QueueFamily::Ignored,
            .nextQueue = RHI::QueueFamily::Ignored,
            .subresourceRange {
                .imageAspect = RHI::Aspect::COLOR_BIT,
                .IndexOrFirstMipLevel = 0,
                .NumMipLevels = 5,
                .FirstArraySlice = 0,
                .NumArraySlices = 6
            }
        }
    };
    ctx.list->PipelineBarrier(RHI::PipelineStage::COMPUTE_SHADER_BIT, RHI::PipelineStage::COMPUTE_SHADER_BIT, {}, img_barr);
    ctx.list->Dispatch(128, 128, 6);
    ctx.list->End();
    ctx.queue->ExecuteCommandLists(&ctx.list->ID, 1);
    ctx.queue->SignalFence(ctx.fence, ++ctx.fence_value);
    ctx.fence->Wait(ctx.fence_value );
}
uint32_t get_offset(const uint32_t level, const uint32_t layer, const uint32_t num_mips,
    const uint32_t width, const uint32_t height, const uint32_t pixel_size)
{
    const uint32_t base_size = pixel_size * width * height;
    const auto mip_chain_size = static_cast<uint32_t>(base_size * (1.0 - std::pow(0.25, num_mips)) / 0.75); //sum of gp
    uint32_t offset = 0;
    const uint32_t num_full_mip_chains_before = layer;
    offset += mip_chain_size * num_full_mip_chains_before;
    offset += static_cast<uint32_t>(base_size * (1.0 - std::pow(0.25, level)) / 0.75);
    return offset;
}
void write_ktx(uint32_t width, uint32_t height, RHI::Weak<RHI::Texture> tex, const char* out, uint32_t num_mips = 1)
{
    ktxTextureCreateInfo inf {
        .glInternalformat = 0,
        .vkFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
        .pDfd = nullptr,
        .baseWidth = width,
        .baseHeight = height,
        .baseDepth = 1,
        .numDimensions = 2,
        .numLevels = 1,
        .numLayers = 1,
        .numFaces = 6,
        .isArray = false,
        .generateMipmaps = false
    };
    ktxTexture2* tx;
    ktxTexture2_Create(&inf, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &tx);
    for(uint32_t i = 0; i < 6; i++){
        const auto data = static_cast<uint8_t*>(tex->Map(RHI::Aspect::COLOR_BIT, 0, i).value());
        const uint32_t face_size = sizeof(uint16_t) * 4 * width * height;
        ktxTexture_SetImageFromMemory(ktxTexture(tx), 0, 0, i, data, face_size);
        tex->UnMap(RHI::Aspect::COLOR_BIT, 0, i);
    }
    ktxTexture_WriteToNamedFile(ktxTexture(tx), out);
}
void generator::generate()
{
    create_base();
    create_ir(32);
    create_pf();

    auto ir_path = output;
    ir_path.erase(ir_path.find_last_of('.'));
    ir_path += "_ir.ktx";

    auto pf_path = output;
    pf_path.erase(pf_path.find_last_of('.'));
    pf_path += "_pf.ktx";
    write_ktx(128, 128, pf, pf_path.c_str(), 5);
    write_ktx(32, 32, ir, ir_path.c_str());
    write_ktx(width, height, base, output.c_str());

}