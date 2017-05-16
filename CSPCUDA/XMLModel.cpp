#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "XMLModel.h"
bool GenerateValues(char* values_str, int values[]);
void GenerateTuples(char *tuple_str, bituple *tuples);
XMLDomain CreateDomain(int id, int size, char* values_str)
{
	XMLDomain domain;
	domain.id = id;
	domain.size = size;
	domain.values = new int[size];
	GenerateValues(values_str, domain.values);

	return domain;
}

XMLVariable CreateVariable(int id, int dm_id)
{
	XMLVariable variable;
	variable.id = id;
	variable.dm_id = dm_id;

	return variable;
}

XMLRelation CreateRelation(int id, int size, int arity, int semantices, char *tuple_str)
{
	XMLRelation relation;
	relation.id = id;
	relation.size = size;
	relation.arity = arity;
	relation.semantices = semantices;
	relation.tuples = new bituple[size];
	GenerateTuples(tuple_str, relation.tuples);

	return relation;
}

XMLConstraint CreateConstraint(int id, int arity, bituple scope, int re_id)
{
	XMLConstraint  constraint;
	constraint.id = id;
	constraint.arity = arity;
	constraint.scope = scope;
	constraint.re_id = re_id;

	return constraint;
}

bool DestroyModel(XMLModel *model)
{
	DestroyDomains(model);
	DestroyVariables(model);
	DestroyRelations(model);
	destroyConstraints(model);
	return true;
}

bool CreateDomains(XMLModel *model, int size)
{
	model->ds_size = size;
	model->domains = new XMLDomain[size];
	return true;
}

bool CreateVariables(XMLModel *model, int size)
{
	model->vs_size = size;
	model->variables = new XMLVariable[size];
	return true;
}

bool CreateRelations(XMLModel *model, int size)
{
	model->rs_size = size;
	model->relations = new XMLRelation[size];
	return true;
}

bool CreateConstraints(XMLModel *model, int size)
{
	model->cs_size = size;
	model->constraints = new XMLConstraint[size];
	return true;
}

bool destroyConstraints(XMLModel *model)
{
	delete[]model->constraints;
	model->constraints = NULL;
	return true;
}

bool DestroyDomains(XMLModel *model)
{
	for (int i = 0; i < model->ds_size; ++i)
	{
		delete[]model->domains[i].values;
		model->domains[i].values = NULL;
	}

	delete[]model->domains;
	model->domains = NULL;

	return true;
}

bool DestroyVariables(XMLModel *model)
{
	delete[]model->variables;
	model->variables = NULL;
	return true;
}

bool DestroyRelations(XMLModel *model)
{
	for (int i = 0; i < model->rs_size; ++i)
	{
		delete[] model->relations[i].tuples;
		model->relations[i].tuples = NULL;
	}

	delete[]model->relations;
	model->relations = NULL;

	return true;
}

void GenerateTuples(char *tuple_str, bituple *tuples)
{
	char* ptr;
	char* context;
	char seps[] = "|";
	int i = 0;

	ptr = strtok_s(tuple_str, seps, &context);

	while (ptr)
	{
		sscanf_s(ptr, "%d %d", &(tuples[i].x), &(tuples[i].y));
		++i;
		ptr = strtok_s(NULL, seps, &context);
	}
}

bool GenerateValues(char* values_str, int *values)
{
	int start = 0;
	int end = 0;
	sscanf_s(values_str, "%d..%d", &start, &end);
	int j = 0;
	for (int i = start; i <= end; ++i)
	{
		values[j] = i;
		++j;
	}

	return true;
}
