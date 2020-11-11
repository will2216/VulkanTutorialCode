
#include <iostream>
#include "Benchmarking.h"

Timer::Timer()
{
	m_startTimepoint = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
	stop();
}

void Timer::stop()
{
	auto endTimepoint = std::chrono::high_resolution_clock::now();

	auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_startTimepoint).time_since_epoch().count();
	auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimepoint).time_since_epoch().count();

	auto duration = end - start;

	double ms = duration * 0.001;

	std::cout << "Function took: " << duration << "\n";
}
