#pragma once
#include "XMLFile.h"


XMLFile::XMLFile(char *file_name) :file_name(file_name)
{
	if (!initial())
	{
		std::cout << "XML initial fail!!" << std::endl;
	}
}


XMLFile::~XMLFile()
{
	if (root)
	{
		delete(parser);
		parser = NULL;
		XMLPlatformUtils::Terminate();
	}
}

bool XMLFile::initial()
{
#pragma region ³õÊ¼»¯Xerces

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
	parser = new XercesDOMParser();
	parser->setValidationScheme(XercesDOMParser::Val_Always);
	parser->setDoNamespaces(true);

	parser->parse(file_name);
	//std::cout << file_name << std::endl;
	document = parser->getDocument();
	root = document->getDocumentElement();

	if (!root)
	{
		delete(parser);
		parser = NULL;
		return false;
	}
#pragma endregion


	return true;
}

char* XMLFile::GetFileName()
{
	DOMNodeList *bmfiles_list = root->getElementsByTagName(XMLString::transcode("BMFiles"));
	DOMNode *bmfiles = bmfiles_list->item(0);
	//DOMNode *bmfiles = root->getFirstChild();
	//std::cout << bmfiles.getNodeName() << std::endl;
	file_length = XMLString::parseInt(bmfiles->getAttributes()->getNamedItem(XMLString::transcode("nbBMFiles"))->getTextContent());
	DOMNodeList *bmfile_list = root->getElementsByTagName(XMLString::transcode("BMFile"));
	DOMNode *bmfile = bmfile_list->item(0);
	char* bm_name = XMLString::transcode(bmfile->getFirstChild()->getNodeValue());
	std::cout << "bm_name = " << bm_name << std::endl;
	//std::cout << root->getChildNodes()->getLength() << std::endl;
	return bm_name;
}

//char** XMLFile::GetFiles()
//{
//	//DOMNodeList *bmfiles_list = root->getElementsByTagName(XMLString::transcode("BenchMark"));
//	DOMNodeList *bmfiles_list = root->getElementsByTagName(XMLString::transcode("BMFiles"));
//	DOMNode *bmfiles = bmfiles_list->item(0);
//	//DOMNode *bmfiles = root->getFirstChild();
//	//std::cout << bmfiles.getNodeName() << std::endl;
//	file_length = XMLString::parseInt(bmfiles->getAttributes()->getNamedItem(XMLString::transcode("nbBMFiles"))->getTextContent());
//
//	std::cout << "file_length = " << file_length << std::endl;
//	//std::cout << root->getChildNodes()->getLength() << std::endl;
//	return NULL;
//}
