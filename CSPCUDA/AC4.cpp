#include "AC4.h"

bool AC4::execute()
{
	initial();
	return true;
}

bool AC4::initial()
{
	int cs_size = model->constraints.size();
	int vs_size = model->variables.size();
	int var0, var1, val0, val1;
	int var0_size;
	int var1_size;
	Constraint *constraint;
	int i, j, k;
	int semantics;
	point tmp_node0;
	point tmp_node1;
	point tmp_node;
	counter_key index_tuple0;
	counter_key index_tuple1;
	counter_key index_tuple_tmp;
	bituple relation_tuple;
	bool find_tuple;
	std::vector<bituple>::iterator t_begin, t_end, t_result;
	std::unordered_map<counter_key, int, KeyHash3, KeyEqual3>::const_iterator counter_start, counter_end;

	for (i = 0; i < cs_size; ++i)
	{
		constraint = &(model->constraints[i]);
		semantics = constraint->relation.semantics;
		var0 = constraint->scope.x;
		var1 = constraint->scope.y;
		var0_size = model->variables[var0].domain.size;
		var1_size = model->variables[var1].domain.size;
		t_begin = constraint->relation.tuples.begin();
		t_end = constraint->relation.tuples.end();

		for (j = 0; j < var0_size; ++j)
		{
			val0 = j;
			tmp_node0 = std::make_tuple(var0, val0);

			for (k = 0; k < var1_size; ++k)
			{
				val1 = k;
				tmp_node1 = std::make_tuple(var1, val1);
				index_tuple0 = std::make_tuple(var0, val0, var1);
				index_tuple1 = std::make_tuple(var1, val1, var0);
				relation_tuple = { val0, val1 };
				t_result = std::find(t_begin, t_end, relation_tuple);
				find_tuple = (t_result != t_end);

				if ((semantics && find_tuple) || (!semantics && !find_tuple))
				{
					++counter[index_tuple0];
					++counter[index_tuple1];
					s[tmp_node1].push_back(tmp_node0);
					s[tmp_node0].push_back(tmp_node1);
				}
				else
				{
					counter[index_tuple0];
					counter[index_tuple1];
				}
			}
		}
	}

	counter_start = counter.begin();
	counter_end = counter.end();
	int counter_len = counter.size();
	for (int i = 0; i < counter_len; ++i)
	{
		if (counter_start->second == 0)
		{
			index_tuple_tmp = counter_start->first;
			var0 = std::get<0>(index_tuple_tmp);
			val0 = std::get<1>(index_tuple_tmp);
			var1 = std::get<2>(index_tuple_tmp);
			std::remove(model->variables[var0].domain.values.begin(), model->variables[var0].domain.values.end(), val0);
			--model->variables[var0].domain.size;
			tmp_node = std::make_tuple(var0, val0);

			if ((model->variables[var0].domain.size) == 0)
			{
				std::cout << "UNSAT!" << std::endl;
				return false;
			}
			q.push_back(tmp_node);
			++delete_nodes_count;
			//----------------------------------
			//std::cout << "(" << std::get<0>(tmp_node) << "," << std::get<1>(tmp_node) << ")" << std::endl;
		}
		++counter_start;
	}
	//------------------------------------
	//std::cout << "initial delete node count = " << delete_nodes_count << std::endl;
	return true;
}

bool AC4::propagate()
{
	point tmp_delete_node;
	point tmp_node;
	counter_key index_tuple;
	std::vector<int>::const_iterator tmp_node_start, tmp_node_end, tmp_result;
	int var0, val0, var1, val1;
	Q *q_e;

	while (!q.empty())
	{
		tmp_delete_node = q.back();
		q.pop_back();
		q_e = &s[tmp_delete_node];
		var1 = std::get<0>(tmp_delete_node);
		val1 = std::get<1>(tmp_delete_node);

		while (!q_e->empty())
		{
			tmp_node = q_e->back();
			q_e->pop_back();
			var0 = std::get<0>(tmp_node);
			val0 = std::get<1>(tmp_node);
			tmp_node_start = model->variables[var0].domain.values.begin();
			tmp_node_end = model->variables[var0].domain.values.end();
			tmp_result = std::find(tmp_node_start, tmp_node_end, val0);

			if (tmp_result == tmp_node_end)
			{
				continue;
			}

			index_tuple = std::make_tuple(var0, val0, var1);
			--counter[index_tuple];
			//std::cout << "counter(" << var0 << "," << val0 << "," << var1 << ") = " << counter[index_tuple] << std::endl;

			if (counter[index_tuple] == 0)
			{
				std::remove(model->variables[var0].domain.values.begin(), model->variables[var0].domain.values.end(), val0);
				--model->variables[var0].domain.size;

				if ((model->variables[var0].domain.size) == 0)
				{
					std::cout << "UNSAT!" << std::endl;
					return false;
				}
				q.push_back(tmp_node);
				++delete_nodes_count;
				//----------------------------
				//std::cout << "(" << std::get<0>(tmp_node) << "," << std::get<1>(tmp_node) << ")" << std::endl;
			}
		}
	}
	//----------------------------------
	//std::cout << "total delete node count = " << delete_nodes_count << std::endl;
	return true;
}