#include <tuple>
#include "Model.h"
typedef std::tuple<int, int> point;

bool Model::GenerateModel()
{
#pragma region 拷贝参数域
	int ds_size = model->ds_size;
	Domains domains(ds_size);
	XMLDomain *xml_domain;

	for (int i = 0; i < ds_size; ++i)
	{
		xml_domain = &(model->domains[i]);
		std::vector<int> values(xml_domain->values, xml_domain->values + xml_domain->size);
		domains[i] = Domain(xml_domain->id, xml_domain->size, values);
	}

	//for (int i = 0; i < domains_size; ++i)
	//{
	//	int  do_size = domains[i].size;

	//	for (int j = 0; j < do_size; ++j)
	//	{
	//		std::cout << domains[i].values[j] << std::endl;
	//	}
	//}
#pragma endregion
#pragma region 拷贝参数
	int vs_size = model->vs_size;
	//variables.resize(vs_size);
	XMLVariable *xml_variable;

	for (int i = 0; i < vs_size; ++i)
	{
		xml_variable = &(model->variables[i]);
		variables[i] = Variable(xml_variable->id, domains[xml_variable->dm_id]);
	}
#pragma endregion
#pragma region 拷贝关系
	int rs_size = model->rs_size;
	Relations relations(rs_size);
	XMLRelation *xml_relation;

	for (int i = 0; i < rs_size; ++i)
	{
		xml_relation = &(model->relations[i]);
		std::vector<bituple> tuples(xml_relation->tuples, xml_relation->tuples + xml_relation->size);
		relations[i] = Relation(xml_relation->id, xml_relation->arity, xml_relation->semantices, xml_relation->size, tuples);
	}
#pragma endregion

#pragma region 拷贝约束
	int cs_size = model->cs_size;
	constraints.resize(cs_size);
	XMLConstraint *xml_constraint;
	for (int i = 0; i < cs_size; i++)
	{
		xml_constraint = &(model->constraints[i]);
		constraints[i] = Constraint(xml_constraint->id, xml_constraint->arity, xml_constraint->scope, relations[xml_constraint->re_id]);
	}
#pragma endregion
	return true;
}
