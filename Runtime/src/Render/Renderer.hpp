#pragma once

#include "Camera/Camera.hpp"
#include "Render/Buffer/Framebuffer.hpp"
#include "Render/Camera/Camera.hpp"
#include "Render/Config/Config.hpp"
#include "Render/RenderPass/RenderPass.hpp"
#include "RenderPass/RenderPass.hpp"
#include "Scene/Scene.hpp"
#include "Shader/Shader.hpp"
#include <cstddef>
#include <glm/fwd.hpp>
#include <memory>
#include <stdint.h>
#include <vector>

namespace suplex {

    enum class RenderType { Forward, Deferred };

    class Renderer {
    public:
        Renderer();
        virtual ~Renderer();
        void OnUpdate(float ts);
        void Render(const std::shared_ptr<Camera> camera, RenderType renderType = RenderType::Forward);
        void PostProcess(const std::shared_ptr<Camera> camera, RenderType renderType = RenderType::Forward);
        void OnUIRender();
        void OnResize(uint32_t w, uint32_t h);

        void OnAwake(bool value)
        {
            for (auto& pass : m_PassQueue)
                pass->Awake(value);
        }

        // uint32_t FramebufferID() const { return m_Framebuffer->GetID(); }
        uint32_t FramebufferImageID() const
        {
            // return m_Framebuffer2->GetTextureID0();
            return m_PostprocessPass->GetFramebufferImage();
        }

        uint32_t DepthMapID() const { return m_DepthPassLS->GetDepthMapID(); }

        uint32_t SceneDepthMapID() const { return m_DepthPass->GetDepthMapID(); }

        float LastFrameRenderTime() const { return m_LastRenderTime; }

        auto& GetGraphicsConfig() { return m_Context->config; }
        auto& GetGraphicsContext() { return m_Context; }
        auto& GetGameObjectList() { return m_Scene; }
        auto& GetShadersList() { return m_ForwardPass->GetShaders(); }
        auto& GetScene() { return m_Scene; }

    private:
        void OnBufferResize();
        void BindRenderPass();
        void BakeEnvironmentLight();

    public:
        uint32_t m_ViewportWidth = 1920, m_ViewportHeight = 1080;

        std::shared_ptr<Framebuffer> m_Framebuffer;

    private:
        std::vector<std::shared_ptr<RenderPass>> m_PassQueue;
        std::shared_ptr<RenderPass>              m_UIRenderPass    = nullptr;
        std::shared_ptr<RenderPass>              m_ForwardPass     = nullptr;
        std::shared_ptr<RenderPass>              m_OutlinePass     = nullptr;
        std::shared_ptr<RenderPass>              m_DepthPassLS     = nullptr;
        std::shared_ptr<RenderPass>              m_DepthPass       = nullptr;
        std::shared_ptr<RenderPass>              m_SSAOPass        = nullptr;
        std::shared_ptr<RenderPass>              m_EnvMapPass      = nullptr;
        std::shared_ptr<RenderPass>              m_PrecomputePass  = nullptr;
        std::shared_ptr<RenderPass>              m_PostprocessPass = nullptr;

        std::shared_ptr<Scene> m_Scene;

        std::shared_ptr<Camera> m_ActiveCamera = nullptr;

        std::shared_ptr<Camera> m_LightCamera = nullptr;

        float m_LastRenderTime = 1.0f;

        std::shared_ptr<GraphicsContext>   m_Context           = nullptr;
        std::shared_ptr<PrecomputeContext> m_PrecomputeContext = nullptr;
    };
}  // namespace suplex