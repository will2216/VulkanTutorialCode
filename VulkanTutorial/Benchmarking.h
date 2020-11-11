#pragma once

#include <chrono>

class Timer
{
public:
	Timer();
	~Timer();

	void stop();

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimepoint;
};

