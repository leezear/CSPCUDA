#pragma once
#include "XMLModelFac.h"
#include "XMLModel.h"


XMLModelFac::XMLModelFac(char *file_name) :file_name(file_name)
{
}


XMLModelFac::~XMLModelFac()
{
}

bool XMLModelFac::GenerateModelFromXml(XMLModel *model)
{
#pragma region 初始化Xerces

	if (this->file_name == NULL)
	{
		return false;
	}

	try
	{
		XMLPlatformUtils::Initialize();
	}
	catch (const XMLException & toCatch)
	{
		// Do your failure processing here  
		return false;
	}
	// Do your actual work with Xerces-C++ here.  
	XercesDOMParser* parser = new XercesDOMParser();
	parser->setValidationScheme(XercesDOMParser::Val_Always);
	parser->setDoNamespaces(true);

	parser->parse(file_name);
	//std::cout << file_name << std::endl;
	xercesc_3_1::DOMDocument *document = parser->getDocument();
	DOMElement *root = document->getDocumentElement();

	if (!root)
	{
		delete(parser);
		parser = NULL;
		return -1;
	}
#pragma endregion

#pragma region 构建域

	DOMNodeList *domains_nodes = root->getElementsByTagName(XMLString::transcode("domains"));
	DOMNode *domains_node = domains_nodes->item(0);
	int domains_count = XMLString::parseInt(domains_node->getAttributes()->getNamedItem(XMLString::transcode("nbDomains"))->getTextContent());
	DOMNodeList *domain_nodes = root->getElementsByTagName(XMLString::transcode("domain"));
	//std::cout << domains_count << std::endl;
	int domain_name;
	int domain_size;
	char *domain_values;
	CreateDomains(model, domains_count);
	for (int i = 0; i < domains_count; ++i)
	{
		domains_node = domain_nodes->item(i);
		domain_name = i;
		domain_size = (int)XMLString::parseInt(domains_node->getAttributes()->getNamedItem(XMLString::transcode("nbValues"))->getTextContent());
		//std::cout << domain_size << " type = " << sizeof(domain_size) << " tpye = " << sizeof(int) << std::endl;
		domain_values = XMLString::transcode(domains_node->getFirstChild()->getNodeValue());
		XMLDomain domain = CreateDomain(domain_name, domain_size, domain_values);
		model->domains[i] = domain;
	}
#pragma endregion

#pragma region 构建参数
	DOMNode *variables_node = root->getElementsByTagName(XMLString::transcode("variables"))->item(0);
	int variables_count = (int)XMLString::parseInt(variables_node->getAttributes()->getNamedItem(XMLString::transcode("nbVariables"))->getTextContent());
	DOMNodeList* variable_nodes = root->getElementsByTagName(XMLString::transcode("variable"));
	//model->GenerateVariables(variables_count);
	CreateVariables(model, variables_count);
	DOMNode* variable_node;
	int variable_name;
	char* domain_name_str;

	for (int i = 0; i < variables_count; ++i)
	{
		variable_name = i;
		variable_node = variable_nodes->item(i);
		domain_name_str = XMLString::transcode(variable_node->getAttributes()->getNamedItem(XMLString::transcode("domain"))->getTextContent());
		sscanf_s(domain_name_str, "D%d", &domain_name);
		model->variables[i] = CreateVariable(variable_name, domain_name);
		//std::cout << model->variables[i].id << std::endl;
	}
#pragma endregion

#pragma region 构建关系

	DOMNode* relations_node = root->getElementsByTagName(XMLString::transcode("relations"))->item(0);
	int relations_count = (int)XMLString::parseInt(relations_node->getAttributes()->getNamedItem(XMLString::transcode("nbRelations"))->getTextContent());
	DOMNodeList* relation_nodes = root->getElementsByTagName(XMLString::transcode("relation"));
	CreateRelations(model, relations_count);
	DOMNode *relation_node;
	char* semantics;
	char* innertext;
	int relation_name;
	int tuple_arity;
	int tuples_count;
	int count;
	int r_type;

	for (int i = 0; i < relations_count; ++i)
	{
		relation_name = i;
		relation_node = relation_nodes->item(i);
		tuple_arity = (int)XMLString::parseInt(relation_node->getAttributes()->getNamedItem(XMLString::transcode("arity"))->getTextContent());

		if (tuple_arity != 2)
		{
			return false;
			std::cout << "输入应为二元约束" << std::endl;
		}

		semantics = XMLString::transcode(relation_node->getAttributes()->getNamedItem(XMLString::transcode("semantics"))->getTextContent());
		tuples_count = XMLString::parseInt(relation_node->getAttributes()->getNamedItem(XMLString::transcode("nbTuples"))->getTextContent());
		//若属性semantics == supports则relation_type = SURPPOT
		r_type = (strlen(semantics) == strlen("supports")) ? 1 : 0;
		innertext = XMLString::transcode(relation_node->getFirstChild()->getNodeValue());
		model->relations[i] = CreateRelation(relation_name, tuples_count, tuple_arity, r_type, innertext);
	}

#pragma endregion

#pragma region 创建约束
	bituple scope;
	int relation_id;
	int constraint_name;
	char *relation_id_str;
	DOMNode* constraint_node;
	int constraint_arity_new;
	char *constraint_scop_str;
	DOMNode* constraints_node = root->getElementsByTagName(XMLString::transcode("constraints"))->item(0);
	int constraints_count = (int)XMLString::parseInt(constraints_node->getAttributes()->getNamedItem(XMLString::transcode("nbConstraints"))->getTextContent());
	DOMNodeList* constraint_nodes = root->getElementsByTagName(XMLString::transcode("constraint"));
	CreateConstraints(model, constraints_count);
	int constraint_arity = (int)XMLString::parseInt(constraint_nodes->item(0)->getAttributes()->getNamedItem(XMLString::transcode("arity"))->getTextContent());

	for (int i = 0; i < constraints_count; ++i)
	{
		constraint_name = i;
		constraint_node = constraint_nodes->item(i);
		constraint_arity_new = (int)XMLString::parseInt(constraint_node->getAttributes()->getNamedItem(XMLString::transcode("arity"))->getTextContent());

		if (constraint_arity_new != 2)
		{
			return false;
			std::cout << "输入应为二元约束" << std::endl;
		}

		constraint_arity = 2;
		constraint_scop_str = XMLString::transcode(constraint_node->getAttributes()->getNamedItem(XMLString::transcode("scope"))->getTextContent());
		sscanf_s(constraint_scop_str, "V%d V%d", &scope.x, &scope.y);
		relation_id_str = XMLString::transcode(constraint_node->getAttributes()->getNamedItem(XMLString::transcode("reference"))->getTextContent());
		sscanf_s(relation_id_str, "R%d", &relation_id);
		model->constraints[i] = CreateConstraint(constraint_name, constraint_arity, scope, relation_id);
	}
#pragma endregion

	delete(parser);
	parser = NULL;
	XMLPlatformUtils::Terminate();

	return true;
}