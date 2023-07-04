#include <MiniEngine.h>

class Sandbox : public ME::Application
{
public:
	Sandbox()
	{

	}
	~Sandbox()
	{

	}
};

ME::Application* ME::CreateApplication()
{
	return new Sandbox();
}