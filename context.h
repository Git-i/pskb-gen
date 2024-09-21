#pragma once
#include "Instance.h"
class context {
public:
    void initialize()
    {
        inst = RHI::Instance::Create().value();
        RHI::PhysicalDevice* dev;
        inst->GetPhysicalDevice(0, &dev);
        auto desc = RHI::CommandQueueDesc{.commandListType = RHI::CommandListType::Direct, .Priority = 1.f};
        std::vector<RHI::Ptr<RHI::CommandQueue>> q_vec;
        std::tie(device, q_vec) = RHI::Device::Create(dev, {&desc,1}, inst).value();
        queue = q_vec[0];
        allocator = device->CreateCommandAllocator(RHI::CommandListType::Direct).value();
        list = device->CreateCommandList(RHI::CommandListType::Direct, allocator).value();
        fence = device->CreateFence(0).value();
        RHI::PoolSize ps[2] {
            {
                .type = RHI::DescriptorType::CSTexture,
               .numDescriptors = 10
            },
            {
                .type = RHI::DescriptorType::SampledTexture,
                .numDescriptors = 10
            }
        };
        descriptorHeap = device->CreateDescriptorHeap(RHI::DescriptorHeapDesc{
            .maxDescriptorSets = 20,
            .numPoolSizes = 2,
            .poolSizes = ps}).value();
        RHI::PoolSize sps{
            .type = RHI::DescriptorType::Sampler,
            .numDescriptors = 10
        };
        samplerHeap = device->CreateDescriptorHeap(RHI::DescriptorHeapDesc{
            .maxDescriptorSets = 10,
            .numPoolSizes = 1,
            .poolSizes = &sps
        }).value();
    }
private:
    friend class generator;
    RHI::Ptr<RHI::Device> device;
    RHI::Ptr<RHI::Instance> inst;
    RHI::Ptr<RHI::CommandQueue> queue;
    RHI::Ptr<RHI::CommandAllocator> allocator;
    RHI::Ptr<RHI::GraphicsCommandList> list;
    RHI::Ptr<RHI::Fence> fence;
    RHI::Ptr<RHI::DescriptorHeap> descriptorHeap;
    RHI::Ptr<RHI::DescriptorHeap> samplerHeap;
    std::uint64_t fence_value = 0;
};