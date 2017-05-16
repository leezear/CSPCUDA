#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct bituple
{
	int x;
	int y;
	bool operator==(const bituple &rhs)
	{
		return (x == rhs.x) && (y == rhs.y);
	}
};

typedef struct XMLDomain
{
	int id;
	int size;
	int* values;
};

typedef struct XMLVariable
{
	int id;
	int dm_id;
};

typedef struct XMLRelation
{
	int id;
	int size;
	int arity;
	int semantices;
	bituple *tuples;
};

typedef struct XMLConstraint
{
	int id;
	int arity;
	bituple scope;
	int re_id;
};

typedef struct XMLModel
{
	int ds_size;
	int vs_size;
	int rs_size;
	int cs_size;
	XMLDomain *domains;
	XMLVariable *variables;
	XMLRelation *relations;
	XMLConstraint *constraints;
};

XMLDomain CreateDomain(int id, int size, char* values_str);
XMLVariable CreateVariable(int id, int dm_id);
XMLRelation CreateRelation(int id, int size, int arity, int semantices, char *tuple_str);
XMLConstraint CreateConstraint(int id, int arity, bituple scope, int re_id);
bool CreateDomains(XMLModel *model, int size);
bool CreateVariables(XMLModel *model, int size);
bool CreateRelations(XMLModel *model, int size);
bool CreateConstraints(XMLModel *model, int size);
XMLModel *CreateModel();
bool DestroyModel(XMLModel *model);
bool DestroyDomains(XMLModel *model);
bool DestroyVariables(XMLModel *model);
bool DestroyRelations(XMLModel *model);
bool destroyConstraints(XMLModel *model);
