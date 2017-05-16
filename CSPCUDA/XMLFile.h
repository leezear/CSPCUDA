#pragma once
#ifndef _DEBUG
#pragma   comment(   lib,   "xerces-c_3.lib"   ) 
#else
#pragma   comment(   lib,   "xerces-c_3D.lib"   ) 
#endif

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMTreeWalker.hpp>
#include <iostream>
#include "XMLModel.h"

using namespace xercesc;

class XMLFile
{
public:
	char *file_name;
	int file_length;
	XercesDOMParser* parser;
	DOMElement *root;
	xercesc_3_1::DOMDocument *document;

	XMLFile(char *file_name);
	~XMLFile();
	char* GetFileName();
	//char** GetFiles();
private:
	bool initial();
};

