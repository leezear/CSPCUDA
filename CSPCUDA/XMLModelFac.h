#pragma once

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMTreeWalker.hpp>
#include <iostream>
#include "XMLModel.h"

using namespace xercesc_3_1;
typedef xercesc_3_1::DOMDocument DOMDOC;

class XMLModelFac
{
public:
	XMLModelFac(char *file_name);
	~XMLModelFac();
	bool GenerateModelFromXml(XMLModel *model);

private:
	char* file_name;
};