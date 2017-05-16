#pragma once
#include <vector>
#include <iostream>
#include "XMLModel.h"
class Domain
{
public:
	Domain(){};
	Domain(int id, int size, std::vector<int> values) :id(id), size(size), values(values){};
	~Domain(){};
	int id;
	int size;
	std::vector<int> values;
};

class Variable
{
public:
	Variable(){};
	Variable(int id, Domain domain) :id(id), domain(domain){};
	~Variable(){};
	int id;
	Domain domain;
};

class Relation
{
public:
	Relation(){};
	Relation(int id, int arity, int semantics, int tuple_count, std::vector<bituple> tuples) :id(id), arity(arity), semantics(semantics), tuple_count(tuple_count), tuples(tuples){};
	~Relation(){};
	int id;
	int arity;
	int semantics;
	int tuple_count;
	std::vector<bituple> tuples;
};

class Constraint
{
public:
	Constraint(){};
	Constraint(int id, int arity, bituple scope, Relation relation) :id(id), arity(arity), scope(scope), relation(relation){};
	~Constraint(){};
	int id;
	int arity;
	bituple scope;
	Relation relation;
};

typedef std::vector<Domain> Domains;
typedef std::vector<Variable> Variables;
typedef std::vector<Relation> Relations;
typedef std::vector<Constraint> Constraints;

class Model
{
public:
	Model(XMLModel *model) :model(model){
		variables.resize(model->vs_size);
		constraints.resize(model->cs_size);
	};
	~Model(){};
	bool GenerateModel();

	Constraints constraints;
	Variables variables;

private:
	XMLModel *model;
};