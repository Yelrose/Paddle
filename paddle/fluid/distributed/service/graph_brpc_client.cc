// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "Eigen/Dense"

#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/table/table.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/string/string_helper.h"
namespace paddle {
namespace distributed {

int GraphBrpcClient::get_server_index_by_id(uint64_t id) {
  int shard_num = get_shard_num();
  int shard_per_server = shard_num % server_size == 0
                             ? shard_num / server_size
                             : shard_num / server_size + 1;
  return id % shard_num / shard_per_server;
}
// char* &buffer,int &actual_size
std::future<int32_t> GraphBrpcClient::sample(uint32_t table_id,
                                             uint64_t node_id, int sample_size,
                                             std::vector<GraphNode> &res) {
  int server_index = get_server_index_by_id(node_id);
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [&](void *done) {
    int ret = 0;
    auto *closure = (DownpourBrpcClosure *)done;
    if (closure->check_response(0, PS_GRAPH_SAMPLE) != 0) {
      ret = -1;
    } else {
      VLOG(0) << "check sample response: "
              << " " << closure->check_response(0, PS_GRAPH_SAMPLE);
      auto &res_io_buffer = closure->cntl(0)->response_attachment();
      butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
      size_t bytes_size = io_buffer_itr.bytes_left();
      char *buffer = new char[bytes_size];
      io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);

      size_t num_nodes = *(size_t *)buffer;
      int *actual_sizes = (int *)(buffer + sizeof(size_t));
      char *node_buffer = buffer + sizeof(size_t) + sizeof(int) * num_nodes;
      
      std::vector<std::vector<GraphNode> > ress;
      std::vector<GraphNode> res_;
      int offset = 0;
      for (size_t idx = 0; idx < num_nodes; ++idx){
        int actual_size = actual_sizes[idx];
        int start = 0;
        while (start < actual_size) {
          GraphNode node;
          node.recover_from_buffer(node_buffer + offset + start);
          start += node.get_size();
          res_.push_back(node);
        }
        offset += actual_size;
        ress.push_back(res_);
      }
      res = ress[0];
    }
    closure->set_promise_value(ret);
  });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  ;
  closure->request(0)->set_cmd_id(PS_GRAPH_SAMPLE);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  // std::string type_str = GraphNode::node_type_to_string(type);
  std::vector<uint64_t> node_ids;
  node_ids.push_back(node_id);
  size_t num_nodes = node_ids.size();
    
  closure->request(0)->add_params((char *)node_ids.data(), sizeof(uint64_t)*num_nodes);
  closure->request(0)->add_params((char *)&sample_size, sizeof(int));
  PsService_Stub rpc_stub(get_cmd_channel(server_index));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);

  return fut;
}

std::future<int32_t> GraphBrpcClient::batch_sample(uint32_t table_id,
                                             std::vector<uint64_t> node_ids, int sample_size,
                                             std::vector<std::vector<GraphNode> > &res) {
  uint64_t node_id = node_ids[0];
  int server_index = get_server_index_by_id(node_id);
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [&](void *done) {
    int ret = 0;
    auto *closure = (DownpourBrpcClosure *)done;
    if (closure->check_response(0, PS_GRAPH_SAMPLE) != 0) {
      ret = -1;
    } else {
      VLOG(0) << "check sample response: "
              << " " << closure->check_response(0, PS_GRAPH_SAMPLE);
      auto &res_io_buffer = closure->cntl(0)->response_attachment();
      butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
      size_t bytes_size = io_buffer_itr.bytes_left();
      char *buffer = new char[bytes_size];
      io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);

      size_t num_nodes = *(size_t *)buffer;
      int *actual_sizes = (int *)(buffer + sizeof(size_t));
      char *node_buffer = buffer + sizeof(size_t) + sizeof(int) * num_nodes;
      
      std::vector<GraphNode> res_;
      int offset = 0;
      std::cout << "num_nodes: " << num_nodes << std::endl;
      for (size_t idx = 0; idx < num_nodes; ++idx){
        int actual_size = actual_sizes[idx];
        std::cout << "actual_size: " << actual_size << std::endl;
        int start = 0;
        while (start < actual_size) {
          GraphNode node;
          node.recover_from_buffer(node_buffer + offset + start);
          start += node.get_size();
          res_.push_back(node);
        }
        offset += actual_size;
        res.push_back(res_);
      }
    }
    closure->set_promise_value(ret);
  });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  ;
  closure->request(0)->set_cmd_id(PS_GRAPH_SAMPLE);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  // std::string type_str = GraphNode::node_type_to_string(type);
  size_t num_nodes = node_ids.size();
    
  closure->request(0)->add_params((char *)node_ids.data(), sizeof(uint64_t)*num_nodes);
  closure->request(0)->add_params((char *)&sample_size, sizeof(int));
  PsService_Stub rpc_stub(get_cmd_channel(server_index));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);

  return fut;
}

std::future<int32_t> GraphBrpcClient::pull_graph_list(
    uint32_t table_id, int server_index, int start, int size,
    std::vector<GraphNode> &res) {
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [&](void *done) {
    int ret = 0;
    auto *closure = (DownpourBrpcClosure *)done;
    if (closure->check_response(0, PS_PULL_GRAPH_LIST) != 0) {
      ret = -1;
    } else {
      VLOG(0) << "check sample response: "
              << " " << closure->check_response(0, PS_PULL_GRAPH_LIST);
      auto &res_io_buffer = closure->cntl(0)->response_attachment();
      butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
      size_t bytes_size = io_buffer_itr.bytes_left();
      char *buffer = new char[bytes_size];
      io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);
      int index = 0;
      while (index < bytes_size) {
        GraphNode node;
        node.recover_from_buffer(buffer + index);
        index += node.get_size();
        res.push_back(node);
      }
    }
    closure->set_promise_value(ret);
  });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  ;
  closure->request(0)->set_cmd_id(PS_PULL_GRAPH_LIST);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  closure->request(0)->add_params((char *)&start, sizeof(int));
  closure->request(0)->add_params((char *)&size, sizeof(int));
  PsService_Stub rpc_stub(get_cmd_channel(server_index));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);
  return fut;
}
int32_t GraphBrpcClient::initialize() {
  set_shard_num(_config.shard_num());
  BrpcPsClient::initialize();
  server_size = get_server_nums();
  return 0;
}
}
}
