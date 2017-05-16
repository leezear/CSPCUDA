#pragma once
#include <tuple>
#include <deque>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <time.h>
#include "Model.h"

typedef std::tuple<int, int> point;
typedef std::tuple<int, int, int> counter_key;
typedef std::deque<point> Q;

struct KeyHash2
{
	int operator()(const point& p)const
	{
		return std::hash<int>()(std::get<0>(p)) ^ (std::hash<int>()(std::get<1>(p)) << 1);
	}

};

struct KeyEqual2
{
	bool operator()(const point &lhs, const point &rhs)const
	{
		return (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs));
	}
};

struct KeyHash3
{
	int operator()(const counter_key& ck)const
	{
		return ((std::hash<int>()(std::get<0>(ck))
			^ (std::hash<int>()(std::get<1>(ck)) << 1)) >> 1)
			^ (std::hash<int>()(std::get<2>(ck)) << 1);
	}

};

struct KeyEqual3
{
	bool operator()(const counter_key &lhs, const counter_key &rhs)const
	{
		return (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs)) && (std::get<2>(lhs) == std::get<2>(rhs));
	}
};

typedef std::unordered_map<point, Q, KeyHash2, KeyEqual2> S;
typedef std::unordered_map<counter_key, int, KeyHash3, KeyEqual3> Counter;

class AC4
{
public:
	S s;
	Counter counter;
	Q q;
	Model *model;
	int delete_nodes_count = 0;
	//float build_time, initial_time, propagate_time;
	AC4(Model *model) :model(model){};
	~AC4(){};
	bool execute();
	bool initial();
	bool propagate();
};




