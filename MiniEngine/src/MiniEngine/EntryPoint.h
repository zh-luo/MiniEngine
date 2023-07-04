#pragma once
//#include "Hazel/Core/Base.h"
#include "MiniEngine/Application.h"

#ifdef ME_PLATFORM_WINDOWS

extern ME::Application* ME::CreateApplication();

int main(int argc, char** argv)
{
	//Hazel::Log::Init();

	//HZ_PROFILE_BEGIN_SESSION("Startup", "HazelProfile-Startup.json");
	auto app = ME::CreateApplication();
	//HZ_PROFILE_END_SESSION();

	//HZ_PROFILE_BEGIN_SESSION("Runtime", "HazelProfile-Runtime.json");
	app->Run();
	//HZ_PROFILE_END_SESSION();

	//HZ_PROFILE_BEGIN_SESSION("Shutdown", "HazelProfile-Shutdown.json");
	delete app;
	//HZ_PROFILE_END_SESSION();
}

#endif
