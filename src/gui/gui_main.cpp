#include "app_state.hpp"
#include "resource.hpp"

#include <imgui.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>
#include <d3d11.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <tchar.h>

#include <algorithm>
#include <filesystem>

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

ID3D11Device*            g_pd3dDevice = nullptr;
ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
IDXGISwapChain*          g_pSwapChain = nullptr;
UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;
fzip::gui::AppState*     g_pAppState = nullptr;

void apply_fzip_theme(float scale) {
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowPadding     = ImVec2(14.0f, 12.0f);
    style.FramePadding      = ImVec2(10.0f, 6.0f);
    style.CellPadding       = ImVec2(10.0f, 6.0f);
    style.ItemSpacing       = ImVec2(10.0f, 8.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);
    style.WindowRounding    = 0.0f;
    style.ChildRounding     = 8.0f;
    style.FrameRounding     = 6.0f;
    style.PopupRounding     = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 6.0f;
    style.TabRounding       = 6.0f;
    style.WindowBorderSize  = 0.0f;
    style.ChildBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 12.0f;
    style.ScaleAllSizes(scale);

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text]                 = ImVec4(0.92f, 0.94f, 0.97f, 1.00f);
    c[ImGuiCol_TextDisabled]         = ImVec4(0.52f, 0.57f, 0.65f, 1.00f);
    c[ImGuiCol_WindowBg]             = ImVec4(0.051f, 0.067f, 0.090f, 1.00f);
    c[ImGuiCol_ChildBg]              = ImVec4(0.082f, 0.106f, 0.137f, 1.00f);
    c[ImGuiCol_PopupBg]              = ImVec4(0.094f, 0.122f, 0.161f, 0.98f);
    c[ImGuiCol_Border]               = ImVec4(0.165f, 0.208f, 0.275f, 1.00f);
    c[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_FrameBg]              = ImVec4(0.114f, 0.145f, 0.192f, 1.00f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(0.153f, 0.196f, 0.259f, 1.00f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(0.184f, 0.239f, 0.318f, 1.00f);
    c[ImGuiCol_TitleBg]              = ImVec4(0.051f, 0.067f, 0.090f, 1.00f);
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.075f, 0.098f, 0.129f, 1.00f);
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.067f, 0.086f, 0.114f, 1.00f);
    c[ImGuiCol_ScrollbarGrab]        = ImVec4(0.220f, 0.275f, 0.361f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.290f, 0.361f, 0.471f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.184f, 0.506f, 0.969f, 1.00f);
    c[ImGuiCol_CheckMark]            = ImVec4(0.345f, 0.651f, 1.000f, 1.00f);
    c[ImGuiCol_SliderGrab]           = ImVec4(0.184f, 0.506f, 0.969f, 1.00f);
    c[ImGuiCol_SliderGrabActive]     = ImVec4(0.345f, 0.651f, 1.000f, 1.00f);
    c[ImGuiCol_Button]               = ImVec4(0.137f, 0.176f, 0.235f, 1.00f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(0.184f, 0.239f, 0.318f, 1.00f);
    c[ImGuiCol_ButtonActive]         = ImVec4(0.145f, 0.392f, 0.780f, 1.00f);
    c[ImGuiCol_Header]               = ImVec4(0.184f, 0.506f, 0.969f, 0.26f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(0.184f, 0.506f, 0.969f, 0.42f);
    c[ImGuiCol_HeaderActive]         = ImVec4(0.184f, 0.506f, 0.969f, 0.58f);
    c[ImGuiCol_Separator]            = ImVec4(0.165f, 0.208f, 0.275f, 1.00f);
    c[ImGuiCol_PlotHistogram]        = ImVec4(0.184f, 0.506f, 0.969f, 1.00f);
    c[ImGuiCol_PlotHistogramHovered] = ImVec4(0.345f, 0.651f, 1.000f, 1.00f);
    c[ImGuiCol_TableHeaderBg]        = ImVec4(0.114f, 0.145f, 0.192f, 1.00f);
    c[ImGuiCol_TableBorderStrong]    = ImVec4(0.165f, 0.208f, 0.275f, 1.00f);
    c[ImGuiCol_TableBorderLight]     = ImVec4(0.125f, 0.161f, 0.212f, 1.00f);
    c[ImGuiCol_TableRowBg]           = ImVec4(0.075f, 0.098f, 0.129f, 1.00f);
    c[ImGuiCol_TableRowBgAlt]        = ImVec4(0.094f, 0.122f, 0.161f, 1.00f);
    c[ImGuiCol_TextSelectedBg]       = ImVec4(0.184f, 0.506f, 0.969f, 0.35f);
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    if (pBackBuffer) {
        g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
        pBackBuffer->Release();
    }
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}

bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
                                               featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                               &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED) {
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags,
                                           featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                           &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    }
    if (res != S_OK) return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            if (wParam == SIZE_MINIMIZED) return 0;
            g_ResizeWidth = (UINT)LOWORD(lParam);
            g_ResizeHeight = (UINT)HIWORD(lParam);
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_DROPFILES: {
            HDROP hDrop = (HDROP)wParam;
            UINT count = DragQueryFileW(hDrop, 0xFFFFFFFF, nullptr, 0);
            std::vector<std::string> paths;
            for (UINT i = 0; i < count; ++i) {
                UINT len = DragQueryFileW(hDrop, i, nullptr, 0);
                if (len > 0) {
                    std::wstring wpath(len + 1, 0);
                    DragQueryFileW(hDrop, i, wpath.data(), len + 1);
                    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
                    if (utf8_len > 1) {
                        std::string spath(utf8_len - 1, 0);
                        WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, spath.data(), utf8_len, nullptr, nullptr);
                        paths.push_back(std::move(spath));
                    }
                }
            }
            DragFinish(hDrop);
            if (g_pAppState && !paths.empty()) {
                g_pAppState->add_input_paths(paths);
            }
            return 0;
        }
        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}

}  // namespace

#if defined(_WIN32) && defined(FZIP_GUI_WINMAIN)
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    int argc = __argc;
    char** argv = __argv;
#else
int main(int argc, char** argv) {
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    int nCmdShow = SW_SHOWDEFAULT;
#endif
    fzip::gui::AppState app;
    g_pAppState = &app;
    app.init(argc, argv);

    if (app.should_close()) {
        return 0;
    }

    ImGui_ImplWin32_EnableDpiAwareness();

    HICON hIconBig = static_cast<HICON>(::LoadImageW(
        hInstance, MAKEINTRESOURCEW(IDI_FZIP_APP), IMAGE_ICON,
        ::GetSystemMetrics(SM_CXICON), ::GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR));
    HICON hIconSmall = static_cast<HICON>(::LoadImageW(
        hInstance, MAKEINTRESOURCEW(IDI_FZIP_APP), IMAGE_ICON,
        ::GetSystemMetrics(SM_CXSMICON), ::GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR));

    // Register Win32 window class
    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
        hInstance, hIconBig, nullptr, nullptr, nullptr,
        L"fzip_gui_wndclass", hIconSmall
    };
    ::RegisterClassExW(&wc);

    RECT wa = { 0, 0, 1280, 800 };
    ::SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
    const int base_w = app.is_quick_mode() ? 580 : 1000;
    const int base_h = app.is_quick_mode() ? 270 : 680;
    const int x0 = wa.left + std::max(0, static_cast<int>((wa.right - wa.left) - base_w) / 2);
    const int y0 = wa.top + std::max(0, static_cast<int>((wa.bottom - wa.top) - base_h) / 2);

    const DWORD wnd_style = app.is_quick_mode()
        ? (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX)
        : WS_OVERLAPPEDWINDOW;
    const wchar_t* wnd_title = app.is_quick_mode()
        ? L"fzip - Processing Archive"
        : L"fzip - High-Ratio Binary Arithmetic & Zstd Archiver";

    HWND hwnd = ::CreateWindowW(wc.lpszClassName, wnd_title,
                                wnd_style, x0, y0, base_w, base_h,
                                nullptr, nullptr, wc.hInstance, nullptr);
    if (hIconBig) {
        ::SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hIconBig));
    }
    if (hIconSmall) {
        ::SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hIconSmall));
    }
    app.set_native_window(hwnd);

    const UINT dpi = ::GetDpiForWindow(hwnd);
    const float dpi_scale = (dpi > 0) ? (static_cast<float>(dpi) / 96.0f) : 1.0f;
    if (dpi_scale > 1.01f) {
        const int sw = static_cast<int>(base_w * dpi_scale);
        const int sh = static_cast<int>(base_h * dpi_scale);
        const int sx = wa.left + std::max(0, static_cast<int>((wa.right - wa.left) - sw) / 2);
        const int sy = wa.top + std::max(0, static_cast<int>((wa.bottom - wa.top) - sh) / 2);
        ::SetWindowPos(hwnd, nullptr, sx, sy, sw, sh, SWP_NOZORDER | SWP_NOACTIVATE);
    }

    // Enable dark title bar on Windows 10/11.
    const BOOL use_dark = TRUE;
    constexpr DWORD kDwmwaImmersiveDarkMode = 20u;
    if (FAILED(::DwmSetWindowAttribute(hwnd, kDwmwaImmersiveDarkMode, &use_dark, sizeof(use_dark)))) {
        constexpr DWORD kDwmwaImmersiveDarkModeLegacy = 19u;
        ::DwmSetWindowAttribute(hwnd, kDwmwaImmersiveDarkModeLegacy, &use_dark, sizeof(use_dark));
    }

    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ::DragAcceptFiles(hwnd, TRUE);
    ::ShowWindow(hwnd, nCmdShow);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;

    apply_fzip_theme(dpi_scale);

    ImFont* bold_font = nullptr;
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        ImFontConfig cfg;
        cfg.OversampleH = 2;
        cfg.OversampleV = 2;
        const char* reg_path = "C:\\Windows\\Fonts\\segoeui.ttf";
        const char* bold_path = "C:\\Windows\\Fonts\\segoeuib.ttf";
        if (fs::exists(reg_path, ec)) {
            io.Fonts->AddFontFromFileTTF(reg_path, 16.0f * dpi_scale, &cfg);
            if (fs::exists(bold_path, ec)) {
                bold_font = io.Fonts->AddFontFromFileTTF(bold_path, 19.0f * dpi_scale, &cfg);
            }
        } else {
            io.Fonts->AddFontDefault(&cfg);
        }
    }
    app.set_fonts(bold_font, dpi_scale);

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    bool done = false;
    while (!done && !app.should_close()) {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                done = true;
            }
        }
        if (done) break;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        app.render();

        ImGui::Render();
        const float clear_color[4] = { 0.051f, 0.067f, 0.090f, 1.00f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0); // VSync
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

#else
// Non-Windows (GLFW + OpenGL3) placeholder
#include <cstdio>

int main(int argc, char** argv) {
    fzip::gui::AppState app;
    app.init(argc, argv);
    std::printf("fzip-gui non-Windows platform stub\n");
    return 0;
}
#endif
