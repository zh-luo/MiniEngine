#pragma once

#include "Core.h"

namespace ME {
	class ME_API Application
	{
	public:
		Application();
		virtual ~Application();

		void Run();
	};

	Application* CreateApplication();
}