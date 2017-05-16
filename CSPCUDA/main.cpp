#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
//#include "XMLModel.h"
#include "XMLModelFac.h"
#include "CUDAModel.cuh"
#include "XMLFile.h"
#include "Model.h"
#include "AC4.h"

extern "C" bool DataTransfer(XMLModel *model);
extern "C" bool BuildArcsModel(XMLModel *model);
extern "C" int AC4gpu();
int main()
{
	//char* file_name = "F:\\Projects\\benchmarks\\composed-75-1-2\\composed-75-1-2-0_ext.xml";
	//char* file_name = "F:\\Projects\\benchmarks\\composed-25-1-2\\composed-25-1-2-0_ext.xml";
	XMLFile files("XMLFile.xml");
	char* file_name = files.GetFileName();
	XMLModel* xml_model = new XMLModel();
	XMLModelFac fac(file_name);
	clock_t start_read = clock();
	fac.GenerateModelFromXml(xml_model);
	clock_t end_read = clock();

	clock_t start_build_model = clock();
	Model model(xml_model);
	model.GenerateModel();
	AC4 ac4(&model);
	clock_t end_build_model = clock();

	clock_t start_initial = clock();
	ac4.initial();
	clock_t end_initial = clock();

	clock_t start_propagate = clock();
	ac4.propagate();
	clock_t end_propagate = clock();


	float read = (float)(end_read - start_read) / 1000;
	float build = (float)(end_build_model - start_build_model) / 1000;
	float initial = (float)(end_initial - start_initial) / 1000;
	float propagate = (float)(end_propagate - start_propagate) / 1000;
	float total = read + build + initial + propagate;

	std::cout << ac4.delete_nodes_count << "\tdelete nodes = " << std::endl;
	std::cout << read << "\tread model time = " << std::endl;
	std::cout << build << "\tbuild model time = " << std::endl;
	std::cout << initial << "\tinitialtime = " << std::endl;
	std::cout << propagate << "\tpropagate = " << std::endl;
	std::cout << initial + propagate << "\tinitialtime & propagatetime = " << std::endl;
	std::cout << total << "\ttotal = " << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
	clock_t start_transfer = clock();
	bool Generate_result = DataTransfer(xml_model);
	clock_t end_transfer = clock();
	clock_t start_build = clock();
	BuildArcsModel(xml_model);
	clock_t end_build = clock();
	clock_t start_execute = clock();
	int sat = AC4gpu();
	clock_t end_execute = clock();

	float transfer_time = (float)(end_transfer - start_transfer) / 1000;
	float build_time = (float)(end_build - start_build) / 1000;
	float execute_time = (float)(end_execute - start_execute) / 1000;

	std::cout << transfer_time << "\ttransfer time = " << std::endl;
	std::cout << build_time << " \tbuild arcs time = " << std::endl;
	std::cout << execute_time << "\tinitialtime & propagatetime = " << std::endl;
	std::cout << execute_time + build_time << "\tbuild & execute time = " << std::endl;
	std::cout << transfer_time + build_time + execute_time << "\ttotal = " << std::endl;
	//printf("sat = %d", sat);
	DestroyModel(xml_model);
	return 0;
}