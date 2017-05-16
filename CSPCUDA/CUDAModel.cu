#pragma once
#include "CUDAModel.cuh"
#include <thrust\reduce.h>
#include <thrust\scan.h>
#include <thrust\transform.h>
#include <thrust\pair.h>
#include <thrust\reduce.h>
#include <thrust\sort.h>
#include <thrust\unique.h>
#include <thrust\functional.h>
#include <thrust\for_each.h>
#include <thrust\system\cuda\execution_policy.h>

extern "C" bool DataTransfer(XMLModel *model);
extern "C" bool BuildArcsModel(XMLModel *model);
#define CSCOUNT 2
typedef thrust::tuple<int, int> Node;
typedef thrust::device_vector<Node> Nodes;
typedef thrust::device_vector<int>::iterator   IntIterator;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> D_ArcIterTuple;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, IntIterator> D_ArcSorcIterTuple;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator> D_CounterIterTuple;
typedef thrust::tuple<IntIterator, IntIterator> D_NodesIterTuple;
typedef thrust::zip_iterator<D_ArcIterTuple> D_ArcTupleIter;
typedef thrust::zip_iterator<D_CounterIterTuple> D_CounterIter;
typedef thrust::zip_iterator<D_NodesIterTuple> D_NodesIter;
typedef thrust::zip_iterator<D_ArcSorcIterTuple> D_ArcSorcIter;
typedef thrust::pair<D_CounterIter, IntIterator> Counter;
const static int MAXTHREADSPERBLOCK = 1024;
const static int MaxThreads = 10240000;
int dbsum;
int stream_size;
thrust::device_vector<int> d_vars_size;
thrust::host_vector<int> h_vars_size;
//点偏移量
thrust::device_vector<int> d_node_global;
thrust::device_vector<int> d_nodes_set;
//thrust::device_vector<int> d_segment_indexes;

//counter偏移量
//thrust::device_vector<int> d_local_counter;
//counter总数
int counter_sum;

__device__ bool d_is_sat;
__device__ bool d_conti;

__global__ void ShowDomains(XMLDomain *domains)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int id = domains[i].id;
	int size = domains[i].size;
	int value = domains[i].values[j];
	printf("id = %3d size = %2d value = %2d\n", id, size, value);
}

__global__ void ShowVariables(XMLVariable *variables)
{
	int i = threadIdx.x;
	int id = variables[i].id;
	int dm_id = variables[i].dm_id;
	printf("id = %2d dm_id = %2d\n", id, dm_id);
}

__global__ void ShowRelations(XMLRelation *relations)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	XMLRelation relation = relations[i];
	int size = relation.size;
	if (j < size)
	{
		int id = relation.id;
		bool type = relation.semantices;
		int x = relation.tuples[j].x;
		int y = relation.tuples[j].y;
		printf("id = %3d type = %2d x = %2d y = %2d\n", id, type, x, y);
	}
}

__global__ void ShowConstraints(XMLConstraint *constraints)
{
	int i = threadIdx.x;
	int id = constraints[i].id;
	int re_id = constraints[i].re_id;
	int x = constraints[i].scope.x;
	int y = constraints[i].scope.y;
	printf("cid = %3d re_id = %2d x = %2d y = %2d\n", id, re_id, x, y);
}

struct D_Arcs
{
	thrust::device_vector<int> d_vars0;
	thrust::device_vector<int> d_vals0;
	thrust::device_vector<int> d_vars1;
	thrust::device_vector<int> d_vals1;
	thrust::device_vector<int> d_sorcs;

	D_Arcs()
	{
	}

	D_Arcs(size_t len)
	{
		resize(len);
	}

	void operator()(size_t len)
	{
		resize(len);
	}

	void resize(size_t len)
	{
		d_vars0.resize(len);
		d_vals0.resize(len);
		d_vars1.resize(len);
		d_vals1.resize(len);
		d_sorcs.resize(len);
	}

	//void erase()

}d_arcs, d_arcs2;

struct D_Vars_Size
{
	thrust::device_vector<int> vars;
	thrust::device_vector<int> sizes;
	D_Vars_Size(){}
	D_Vars_Size(size_t len)
	{
		resize(len);
	}

	void resize(size_t len)
	{
		vars.resize(len);
		sizes.resize(len);
	}
};

struct H_Arcs
{
	thrust::host_vector<int> h_vars0;
	thrust::host_vector<int> h_vals0;
	thrust::host_vector<int> h_vars1;
	thrust::host_vector<int> h_vals1;
	thrust::host_vector<int> h_sorcs;

	void operator= (D_Arcs das)
	{
		h_vars0 = das.d_vars0;
		h_vals0 = das.d_vals0;
		h_vars1 = das.d_vars1;
		h_vals1 = das.d_vals1;
		h_sorcs = das.d_sorcs;
	}
}h_arcs;

struct D_Node
{
	thrust::device_vector<int> vars;
	thrust::device_vector<int> vals;
	D_Node()
	{
	}

	D_Node(size_t len)
	{
		vars.resize(len);
		vals.resize(len);
	}

	void operator()(size_t len)
	{
		resize(len);
	}

	void resize(size_t len)
	{
		vars.resize(len);
		vals.resize(len);
	}
}d_nodes;

struct 	D_SegmentIndex
{
	thrust::device_vector<int> start;
	thrust::device_vector<int> end;

	D_SegmentIndex()
	{
	}

	D_SegmentIndex(size_t len)
	{
		resize(len);
	}

	D_SegmentIndex(size_t len, int init_value)
	{
		resize(len, init_value);
	}

	void resize(size_t len)
	{
		start.resize(len);
		end.resize(len);
	}

	void resize(size_t len, int init_value)
	{
		start.resize(len, init_value);
		end.resize(len, init_value);
	}
}d_segidx;

__global__ void ComputeLocalOffset(int *d_offset, int *d_local_counter, XMLConstraint *d_c, XMLDomain *d_d, XMLVariable *d_v)
{
	int i = threadIdx.x;
	int x = d_c[i].scope.x;
	int y = d_c[i].scope.y;
	int x_dm_id = d_v[x].dm_id;
	int y_dm_id = d_v[y].dm_id;
	int x_dm_size = d_d[x_dm_id].size;
	int y_dm_size = d_d[y_dm_id].size;
	int local_offset = x_dm_size*y_dm_size;
	d_local_counter[i] = x_dm_size + y_dm_size;
	d_offset[i] = 2 * local_offset;
}

__global__ void GenerateVars_size(int *vars_size, XMLVariable *vars, XMLDomain *dms)
{
	int i = threadIdx.x;
	int dm_id = vars[i].dm_id;
	int dm_size = dms[dm_id].size;
	vars_size[i] = dm_size;
}

__global__ void GenerateNodes(int *node_var, int *node_val, int *offset, int var_id)
{
	int i = threadIdx.x;
	int idx = offset[var_id] + i;
	node_var[idx] = var_id;
	node_val[idx] = i;
}

__global__ void GenerateNodesLaunch(int *node_var, int *node_val, int *offset, XMLVariable *vars, XMLDomain *dms)
{
	int i = threadIdx.x;
	int dm_id = vars[i].dm_id;
	int dm_size = dms[dm_id].size;
	GenerateNodes << <1, dm_size >> >(node_var, node_val, offset, i);
}

__global__
void BuildArc(
int *d_vars0,
int *d_vals0,
int *d_vars1,
int *d_vals1,
int* d_sorc,
int d_offset,
int d_global_offset,
XMLDomain dm_0,
XMLDomain dm_1,
XMLVariable var_0,
XMLVariable var_1,
XMLConstraint cst, int sorc
)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	int val0 = dm_0.values[x];
	int val1 = dm_1.values[y];
	int var0 = var_0.id;
	int var1 = var_1.id;
	int local_arc0 = blockDim.x*x + y;
	int local_arc1 = gridDim.x*y + x;
	int global_arc0 = local_arc0 + d_global_offset;
	int global_arc1 = local_arc1 + d_offset / 2 + d_global_offset;
	d_vars0[global_arc0] = var0;
	d_vals0[global_arc0] = val0;
	d_vars1[global_arc0] = var1;
	d_vals1[global_arc0] = val1;
	d_vars0[global_arc1] = var1;
	d_vals0[global_arc1] = val1;
	d_vars1[global_arc1] = var0;
	d_vals1[global_arc1] = val0;
	d_sorc[global_arc0] = sorc;
	d_sorc[global_arc1] = sorc;

	return;
}

__global__
void ModifyTuple(
int* d_sorc,
int d_offset,
int d_global_offset,
XMLDomain dm_0,
XMLDomain dm_1,
XMLRelation rel,
int semantices,
int size
)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < size)
	{
		bituple bt = rel.tuples[i];
		int var0_size = dm_0.size;
		int var1_size = dm_1.size;
		int sorc = semantices;
		int local_sorc0 = bt.x*var1_size + bt.y;
		int local_sorc1 = bt.y*var0_size + bt.x;
		int global_sorc0 = local_sorc0 + d_global_offset;
		int global_sorc1 = local_sorc1 + d_offset / 2 + d_global_offset;
		d_sorc[global_sorc0] = sorc;
		d_sorc[global_sorc1] = sorc;
	}
}

__global__
void BuildArcsLaunch(
int *d_vars0,
int *d_vals0,
int *d_vars1,
int *d_vals1,
int *d_sorc,
int *d_offset,
int *d_global_offset,
XMLDomain *dms,
XMLVariable *vars,
XMLRelation *rels,
XMLConstraint *csts
)
{
	int mtpb = 1024;
	int i = threadIdx.x;
	XMLConstraint constraint = csts[i];
	int x = constraint.scope.x;
	int y = constraint.scope.y;
	XMLRelation relation = rels[constraint.re_id];
	int r_size = relation.size;
	XMLVariable var0 = vars[x];
	XMLVariable var1 = vars[y];
	int x_dm_id = var0.dm_id;
	int y_dm_id = var1.dm_id;
	XMLDomain dm_0 = dms[x_dm_id];
	XMLDomain dm_1 = dms[y_dm_id];
	int x_size = dm_0.size;
	int y_size = dm_1.size;
	int semantices = relation.semantices;
	int sorc = !semantices;
	int block_size = r_size / (mtpb)+!(!(r_size % (mtpb)));
	BuildArc << <x_size, y_size >> >(d_vars0, d_vals0, d_vars1, d_vals1, d_sorc, d_offset[i], d_global_offset[i], dm_0, dm_1, var0, var1, constraint, sorc);
	__syncthreads();
	ModifyTuple << <block_size, mtpb >> >(d_sorc, d_offset[i], d_global_offset[i], dm_0, dm_1, relation, semantices, r_size);
}

__global__ void Vars_Resize(int *var, int *del, int*vars)
{
	int i = threadIdx.x;
	int del_var_id = var[i];
	int del_now = del[i];
	int size = vars[del_var_id];
	int new_size = size - del_now;
	vars[del_var_id] = new_size;
}

__global__ void show(int *a)
{
	int i = threadIdx.x;
	printf("%d\n", a[i]);
}

__global__ void CompareVar_Size(int *old_var_size, int *tmp_var_size, int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	d_is_sat = true;
	d_conti = false;

	if (idx < size)
	{
		int old = old_var_size[idx];
		int tmp = tmp_var_size[idx];
		if (old != tmp)
		{
			d_conti = true;
		}
		if (tmp == 0)
		{
			d_is_sat = false;
		}
	}
}

struct TernaryPredicate
{
	template<typename Tuple>
	__host__ __device__ bool operator()(const Tuple& a, const Tuple& b)
	{
		return(
			(thrust::get<0>(a) == thrust::get<0>(b)) &&
			(thrust::get<1>(a) == thrust::get<1>(b)) &&
			(thrust::get<2>(a) == (thrust::get<2>(b)))
			);
	}
};

struct is_conflict
{
	//template<typename Tuple>
	__host__ __device__ bool operator()(const thrust::tuple<const int&, const int&, const int&, const int&, const int&> &a)
	{
		return (!(thrust::get<4>(a)));
	}
};

struct is_zero
{
	int *nodes;
	int *offset;
	int len;
	is_zero(int *nodes, int *offset, int len) :len(len), nodes(nodes), offset(offset){}
	is_zero(){}
	__host__ __device__
		bool operator()(const int s_val) const {
			return (!s_val);
		}
};

struct ModifyNodes
{
	int *nodes;
	int *node_offset;
	ModifyNodes(int *nodes, int *node_offset) :nodes(nodes), node_offset(node_offset){}

	template<typename Tuple>
	__host__ __device__	void operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		int sorc = thrust::get<2>(t);
		if (sorc == 0)
		{
			int node_idx = node_offset[var] + val;
			nodes[node_idx] = 0;
		}
	}
};

struct ModifyArcs
{
	int *nodes;
	int *node_offset;
	int idx;
	ModifyArcs(int *nodes, int *node_offset, int i) :nodes(nodes), node_offset(node_offset), idx(i){}

	template<typename Tuple>
	__host__ __device__	void operator()(const Tuple &t)
	{
		int var0 = thrust::get<0>(t);
		int val0 = thrust::get<1>(t);
		int var1 = thrust::get<2>(t);
		int val1 = thrust::get<3>(t);
		bool del_node = !(nodes[node_offset[var0] + val0] && nodes[node_offset[var1] + val1]);

		if (del_node)
		{
			thrust::get<4>(t) = 0;
			return;
		}
	}
};

struct DeleteNodes
{
	int *nodes;
	int *offset;
	int len;

	DeleteNodes(int *nodes, int *offset, int len) :len(len), nodes(nodes), offset(offset){}

	template <typename Tuple>
	__host__ __device__ void operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		nodes[offset[var] + val] = 0;
	}
};

struct Is_Deleted
{
	int *nodes;
	int *offset;
	int len;
	Is_Deleted(int *nodes, int *offset, int len) : nodes(nodes), offset(offset), len(len) {}

	template <typename Tuple>
	__host__ __device__ bool operator()(const Tuple &t)
	{
		int var = thrust::get<0>(t);
		int val = thrust::get<1>(t);
		int global_offset = offset[var] + val;
		bool deleted = nodes[global_offset];
		return !deleted;
	}
};

struct Build_Segment_index
{
	int *seg;
	int *g_offset;
	int *start;
	int *end;
	int arcdim;
	Build_Segment_index(int *seg, int *g_offset, int arcdim) :seg(seg), g_offset(g_offset), arcdim(arcdim){}
	Build_Segment_index(int *start, int *end, int *g_offset, int arcdim) :start(start), end(end), g_offset(g_offset), arcdim(arcdim){}
	__host__ __device__ void operator()(const int &idx)
	{
		int offset = g_offset[idx];
		int offset_pre = g_offset[idx - 1];
		int segment = offset / arcdim;
		int segment_pre = offset_pre / arcdim;

		if (segment != segment_pre)
		{
			start[segment] = offset_pre;

			if ((segment - 1) >= 0)
			{
				end[segment - 1] = offset_pre - 1;
			}
		}
	}
};

XMLDomain *h_dms;
XMLDomain *d_dms;
XMLVariable *d_vs;
XMLRelation *h_rs;
XMLRelation *d_rs;
XMLConstraint *d_cs;

extern "C" bool DataTransfer(XMLModel *model)
{
#pragma region 拷贝参数域
	int ds_size = model->ds_size;
	int ds_len = ds_size *sizeof(XMLDomain);
	int d_size;
	h_dms = new XMLDomain[ds_size];
	memcpy(h_dms, model->domains, ds_len);

	for (size_t i = 0; i < ds_size; ++i)
	{
		d_size = model->domains[i].size;
		cudaMalloc(&(h_dms[i].values), d_size*sizeof(int));
		cudaMemcpy(h_dms[i].values, model->domains[i].values, d_size*sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void**)&d_dms, ds_len);
	cudaMemcpy(d_dms, h_dms, ds_len, cudaMemcpyHostToDevice);
	//ShowDomains << <ds_size, d_size >> >(d_dms);
#pragma endregion

#pragma region 拷贝参数数组
	int vs_size = model->vs_size;
	int vs_len = vs_size*sizeof(XMLVariable);
	//int v_size;
	cudaMalloc((void **)&d_vs, vs_len);
	cudaMemcpy(d_vs, model->variables, vs_len, cudaMemcpyHostToDevice);
	//ShowVariables << <1, vs_size >> >(d_vs);
#pragma endregion

#pragma region 拷贝关系数组
	int rs_size = model->rs_size;
	int rs_len = rs_size*sizeof(XMLRelation);
	int r_size;
	int r_maxsize = 0;
	int r_len;
	h_rs = new XMLRelation[rs_size];
	memcpy(h_rs, model->relations, rs_len);

	for (size_t i = 0; i < rs_size; i++)
	{
		r_size = model->relations[i].size;
		r_maxsize = (r_size>r_maxsize) ? r_size : r_maxsize;
		r_len = r_size*sizeof(bituple);
		cudaMalloc((void**)&(h_rs[i].tuples), r_len);
		cudaMemcpy(h_rs[i].tuples, model->relations[i].tuples, r_len, cudaMemcpyHostToDevice);
	}
	cudaMalloc((void **)&d_rs, rs_len);
	cudaMemcpy(d_rs, h_rs, rs_len, cudaMemcpyHostToDevice);
	//ShowRelations << <rs_size, r_maxsize >> >(d_rs);
#pragma endregion

#pragma region 拷贝约束
	int cs_size = model->cs_size;
	int cs_len = cs_size*sizeof(XMLConstraint);
	cudaMalloc((void **)&d_cs, cs_len);
	cudaMemcpy(d_cs, model->constraints, cs_len, cudaMemcpyHostToDevice);
	//ShowConstraints << <1, cs_size >> >(d_cs);
#pragma endregion

#pragma region 构建模型
	//cudaMemcpyToSymbol(d_ds, &ds_size, sizeof(int));
	//cudaMemcpyToSymbol(d_vs, &vs_size, sizeof(int));
	//	cudaMemcpyToSymbol
	//	cudaMemcpyToSymbol
	//cudaMemcpyToSymbol(&d_csize, &model->cs_size, sizeof(int));
	//ShowDeviceVariables << <1, 1 >> >(1);
	//XMLModel *h_model = new XMLModel;
	//XMLModel *d_model;
	//memcpy(h_model, model, sizeof(XMLModel));
	//cudaMalloc((void **)&d_model, sizeof(XMLModel));
	//cudaMalloc((void**)&h_model->domains, sizeof(h_dms));
	//cudaMalloc((void **)&h_model->variables, sizeof(d_vs));
	//cudaMalloc((void**)&h_model->variables, sizeof(h_rs));
	//cudaMalloc((void **)h_model->constraints, sizeof(d_cs));
	//cudaMemcpy(h_model->domains, h_dms, ds_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(h_model->variables, model->variables, vs_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(h_model->relations, h_rs, rs_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(h_model->constraints, model->constraints, cs_len, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_model, h_model, sizeof(XMLModel), cudaMemcpyHostToDevice);
	//ShowModel << <1, 1 >> >(d_model);
#pragma endregion
	return true;
}

extern "C" bool BuildArcsModel(XMLModel *model)
{
	int ds_size = model->ds_size;
	int rs_size = model->rs_size;
	int cs_size = model->cs_size;
	int vs_size = model->vs_size;
#pragma region 计算弧局部/全局偏移量
	thrust::device_vector<int> d_local_counter(cs_size);
	int *d_local_counter_ptr = thrust::raw_pointer_cast(d_local_counter.data());
	thrust::device_vector<int> d_offset(cs_size, 0);
	thrust::device_vector<int> d_global_offset(cs_size, 0);
	thrust::device_vector<int> d_offset_index(cs_size);
	thrust::sequence(d_offset_index.begin(), d_offset_index.end());

	int* d_offset_ptr = thrust::raw_pointer_cast(d_offset.data());
	ComputeLocalOffset << <1, cs_size >> >(d_offset_ptr, d_local_counter_ptr, d_cs, d_dms, d_vs);

	int sum = thrust::reduce(d_offset.begin(), d_offset.end(), (int)0, thrust::plus<int>());
	counter_sum = thrust::reduce(d_local_counter.begin(), d_local_counter.end());
	d_arcs2.resize(counter_sum);

	//printf("counter_sum = %3d", counter_sum);

	dbsum = sum;
	printf("edge = %d\n", dbsum / 2);
	int *d_global_offset_ptr = thrust::raw_pointer_cast(d_global_offset.data());
	thrust::exclusive_scan(d_offset.begin(), d_offset.end(), d_global_offset.begin());

	d_arcs(dbsum);
#pragma endregion

#pragma region 创建var_size数组
	d_vars_size.resize(vs_size);
	int* d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	//var_size *d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	GenerateVars_size << <1, vs_size >> >(d_vars_size_ptr, d_vs, d_dms);
	//h_vars_size = d_vars_size;
#pragma endregion

#pragma region 创建arcs数组
	int *d_vars0_ptr = thrust::raw_pointer_cast(d_arcs.d_vars0.data());
	int *d_vals0_ptr = thrust::raw_pointer_cast(d_arcs.d_vals0.data());
	int *d_vars1_ptr = thrust::raw_pointer_cast(d_arcs.d_vars1.data());
	int *d_vals1_ptr = thrust::raw_pointer_cast(d_arcs.d_vals1.data());
	int *d_sorcs_ptr = thrust::raw_pointer_cast(d_arcs.d_sorcs.data());

	BuildArcsLaunch << <1, cs_size >> >(
		d_vars0_ptr,
		d_vals0_ptr,
		d_vars1_ptr,
		d_vals1_ptr,
		d_sorcs_ptr,
		d_offset_ptr,
		d_global_offset_ptr,
		d_dms,
		d_vs,
		d_rs,
		d_cs
		);

	//int startindex = 0;
	//int endindex = 0;
	//h_arcs = d_arcs;

	//std::cout << "input index range:" << std::endl;
	//scanf("%d %d", &startindex, &endindex);
	//while (!((startindex == -1) && (endindex == -1)))
	//{
	//	if ((startindex == 0) && (endindex == 0))
	//	{
	//		for (size_t i = 0; i < dbsum; ++i)
	//		{
	//			printf("%4d:(%d,%d)--(%d,%d)=%d\n",
	//				i,
	//				h_arcs.h_vars0[i],
	//				h_arcs.h_vals0[i],
	//				h_arcs.h_vars1[i],
	//				h_arcs.h_vals1[i],
	//				h_arcs.h_sorcs[i]
	//				);
	//		}
	//	}
	//	for (size_t i = startindex; i < endindex; ++i)
	//	{
	//		printf("%4d:(%d,%d)--(%d,%d)=%d\n",
	//			i,
	//			h_arcs.h_vars0[i],
	//			h_arcs.h_vals0[i],
	//			h_arcs.h_vars1[i],
	//			h_arcs.h_vals1[i],
	//			h_arcs.h_sorcs[i]);
	//	}
	//	std::cout << "input index range:" << std::endl;
	//	scanf("%d %d", &startindex, &endindex);
	//}
#pragma endregion

#pragma region 计算点局部/全局偏移量
	int nodes_sum = thrust::reduce(d_vars_size.begin(), d_vars_size.end());
	printf("node = %d\n", nodes_sum);
	d_node_global.resize(vs_size);
	thrust::exclusive_scan(d_vars_size.begin(), d_vars_size.end(), d_node_global.begin());
	d_nodes_set.resize(nodes_sum, 1);
	d_nodes.resize(nodes_sum);
	int *var_ptr = thrust::raw_pointer_cast(d_nodes.vars.data());
	int *val_ptr = thrust::raw_pointer_cast(d_nodes.vals.data());
	int *nodes_offset = thrust::raw_pointer_cast(d_node_global.data());
	GenerateNodesLaunch << <1, vs_size >> >(var_ptr, val_ptr, nodes_offset, d_vs, d_dms);
#pragma endregion

#pragma region 释放显存/内存变量
	for (size_t i = 0; i < ds_size; i++)
	{
		cudaFree(h_dms[i].values);
	}

	cudaFree(d_dms);
	delete[]h_dms;
	h_dms = NULL;

	cudaFree(d_vs);

	for (size_t i = 0; i < rs_size; i++)
	{
		cudaFree(h_rs[i].tuples);
	}
	cudaFree(d_rs);
	delete[] h_rs;
	h_rs = NULL;
#pragma endregion

	return true;
}
extern "C" int AC4gpu()
{
#pragma region 初始化数据结构
	int i, j, k, m, n;
	int nodes_sum = d_nodes_set.size();
	int *nodes = thrust::raw_pointer_cast(d_nodes_set.data());
	int *nodes_offset = thrust::raw_pointer_cast(d_node_global.data());
	thrust::device_vector<int> d_vars_size_tmp = d_vars_size;
	thrust::device_vector<int> d_vars_size_key = d_vars_size_tmp;
	cudaStream_t cs[CSCOUNT];
	Counter counter;
	D_CounterIter counter_iter_new;

	int delete_nodes_pre = nodes_sum;
	int delete_nodes_tmp;
	int vs_size = d_vars_size_tmp.size();
	int *d_vars_size_key_ptr;
	int *d_vars_size_key_tmp_ptr;
	int *d_vars_size_ptr;
	d_vars_size_ptr = thrust::raw_pointer_cast(d_vars_size.data());
	d_vars_size_key_ptr = thrust::raw_pointer_cast(d_vars_size_key.data());
	d_vars_size_key_tmp_ptr = thrust::raw_pointer_cast(d_vars_size_tmp.data());
	int block_size = vs_size / (MAXTHREADSPERBLOCK)+!(!(vs_size % (MAXTHREADSPERBLOCK)));
	//printf("vs_size = %d, block_soze  = %d\n", vs_size, block_size);
	//cntn = false  网络sat退出， = -1 unsat， = 1需要下一次迭代
	bool is_sat = true, conti = true;

	for (i = 0; i < CSCOUNT; ++i)
	{
		cudaStreamCreate(&(cs[i]));
	}

#pragma endregion

	while (is_sat&&conti)
	{
		counter = thrust::reduce_by_key(
			thrust::cuda::par.on(cs[0]),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.begin(), d_arcs.d_vals0.begin(), d_arcs.d_vars1.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.end(), d_arcs.d_vals0.end(), d_arcs.d_vars1.end())),
			d_arcs.d_sorcs.begin(),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.begin(), d_arcs2.d_vals0.begin(), d_arcs2.d_vars1.begin())),
			d_arcs2.d_sorcs.begin()
			);

		//待用变长规约
		//int new_len = counter.second - d_arcs2.d_sorcs.begin();
		//printf("new_len = %d", new_len);

		thrust::for_each(
			thrust::cuda::par.on(cs[0]),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.begin(), d_arcs2.d_vals0.begin(), d_arcs2.d_sorcs.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs2.d_vars0.end(), d_arcs2.d_vals0.end(), d_arcs2.d_sorcs.end())),
			ModifyNodes(nodes, nodes_offset)
			);

		cudaStreamSynchronize(cs[0]);

		thrust::reduce_by_key(
			thrust::cuda::par.on(cs[1]),
			d_nodes.vars.begin(),
			d_nodes.vars.end(),
			d_nodes_set.begin(),
			d_vars_size_key.begin(),
			d_vars_size_tmp.begin()
			);

		//比较两次迭代间变化
		CompareVar_Size << <block_size, MAXTHREADSPERBLOCK, 0, cs[1] >> >(d_vars_size_ptr, d_vars_size_key_tmp_ptr, vs_size);
		cudaMemcpyFromSymbolAsync(&is_sat, d_is_sat, sizeof(bool), 0, cudaMemcpyDeviceToHost, cs[1]);
		cudaMemcpyFromSymbolAsync(&conti, d_conti, sizeof(bool), 0, cudaMemcpyDeviceToHost, cs[1]);
		cudaStreamSynchronize(cs[1]);

		d_vars_size = d_vars_size_tmp;

		if (!is_sat)
		{
			std::cout << "UNSAT!!" << std::endl;
			break;
		}
		if (!conti)
		{
			std::cout << "SAT" << std::endl;
			break;
		}

		thrust::for_each(
			thrust::cuda::par.on(cs[0]),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.begin(), d_arcs.d_vals0.begin(), d_arcs.d_vars1.begin(), d_arcs.d_vals1.begin(), d_arcs.d_sorcs.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_arcs.d_vars0.end(), d_arcs.d_vals0.end(), d_arcs.d_vars1.end(), d_arcs.d_vals1.end(), d_arcs.d_sorcs.end())),
			ModifyArcs(nodes, nodes_offset, i)
			);

		cudaStreamSynchronize(cs[0]);
	}

	//for (i = 0; i < nodes_sum; ++i)
	//{
	//	int x = d_nodes.vars[i];
	//	int y = d_nodes.vals[i];
	//	int remain = d_nodes_set[i];

	//	if (remain == 0)
	//	{
	//		std::cout << "(" << x << "," << y << ") = " << remain << std::endl;
	//	}
	//}
#pragma region 释放堆
	for (i = 0; i < CSCOUNT; ++i)
	{
		cudaStreamDestroy(cs[i]);
	}
#pragma endregion

	return 1;
}