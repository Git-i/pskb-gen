#include "generator.h"
#include "ktx.h"
#include "vkformat.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdexcept>
#include <memory>
#include <ranges>
#include "float16.h"
#include "rhi_sc.h"

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
void generator::create_base()
{
    auto& device = ctx.device;
    RHI::AutomaticAllocationInfo inf = {RHI::AutomaticAllocationCPUAccessMode::Random};
    width = 512; height = 512;
    base = device->CreateTexture(RHI::TextureDesc{
            .type = RHI::TextureType::Texture2D,
            .width = width,
            .height = height,
            .depthOrArraySize = 6,
            .format = RHI::Format::R16G16B16A16_FLOAT,
            .mipLevels = 1,
            .sampleCount = 1,
            .mode = RHI::TextureTilingMode::Linear,
            .optimizedClearValue = nullptr,
            .usage = RHI::TextureUsage::CubeMap | RHI::TextureUsage::SampledImage | RHI::TextureUsage::CopyDst | RHI::TextureUsage::StorageImage
        }, nullptr, nullptr, &inf, 0, RHI::ResourceType::Automatic).value();
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


        const auto mapped = static_cast<uint8_t*>(base->Map().value());
        const uint32_t face_size = (width * height * sizeof(uint16_t) * 4);
        size_t off = 0;
        for(auto & i : data)
        {
            auto ptr = image_4ch_half(static_cast<float*>(i), width, height);
            memcpy(mapped + off, ptr.get(), face_size);
            off += face_size;
        }
        base->UnMap();
    }
    else {
        auto cmp = RHI_SC::Compiler::New();
        auto opt = RHI_SC::CompileOptions::New();
        std::vector<char> out;
        if(const auto [warning_count, messages, error] = cmp->CompileToBuffer(ctx.inst->GetInstanceAPI(), RHI_SC::ShaderSource{.source = "../shaders/eq_to_cubemap.hlsl", .stage = RHI::ShaderStage::Compute}, opt, out); error != RHI_SC::CompilationError::None)
            throw std::runtime_error(messages);
        int x, y, nc;
        void* data = stbi_loadf(input.c_str(), &x, &y, &nc, 4);
        width = 512, height = 512;
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

        const auto eq_ptr = static_cast<uint8_t*>(tex->Map().value());
        memcpy(eq_ptr, data, x * y * sizeof(float) * 4);
        tex->UnMap();


        RHI::Ptr<RHI::DescriptorSetLayout> dsl;
        std::array ranges = {
            RHI::DescriptorRange{
                .numDescriptors = 1,
                .BaseShaderRegister = 0,
                .stage = RHI::ShaderStage::Compute,
                .type = RHI::DescriptorType::CSTexture
            },
            RHI::DescriptorRange{
                .numDescriptors = 1,
                .BaseShaderRegister = 1,
                .stage = RHI::ShaderStage::Compute,
                .type = RHI::DescriptorType::SampledTexture
            },
            RHI::DescriptorRange{
                .numDescriptors = 1,
                .BaseShaderRegister = 2,
                .stage = RHI::ShaderStage::Compute,
                .type = RHI::DescriptorType::Sampler
            },
        };
        std::array rp_desc = {
            RHI::RootParameterDesc{
            .type = RHI::RootParameterType::DescriptorTable,
            .descriptorTable {
                .ranges = ranges,
                .setIndex = 0
            }}
        };
        auto rs = device->CreateRootSignature(RHI::RootSignatureDesc{rp_desc}, &dsl).value();
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
        device->UpdateDescriptorSet(3, updates, set);
        ctx.list->Begin(ctx.allocator);
        std::array img_barr = {
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
void generator::create_ir()
{

}
void generator::create_pf()
{

}
void generator::generate()
{
    create_base();
    create_ir();
    create_pf();

    

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
    const auto data = static_cast<uint8_t*>(base->Map().value());
    for(uint32_t i = 0; i < 6; i++){
        const uint32_t face_size = (width * height * sizeof(uint16_t) * 4);
        ktxTexture_SetImageFromMemory(ktxTexture(tx), 0, 0, i, data + face_size * i, face_size);
    }
    base->UnMap();
    ktxTexture_WriteToNamedFile(ktxTexture(tx), output.c_str());
}