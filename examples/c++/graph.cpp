#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stdlib.h"
#include <vector> 
#include <unordered_set>
#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <memory>

#define MAX_NUM_OPS 8
using namespace std;

enum class OpType {
    Add,
    Subtract,
    Multiply,
    Divide,
    // Add more operation types here
};

void print_dense_attr(mlir::DenseElementsAttr attr) {
    // Get the type of elements stored in the attribute
    auto type = attr.getType().cast<mlir::ShapedType>();
    
    // Check the element type and print accordingly
    if (type.getElementType().isF32()) {
        // For floating point elements, print each value
        for (float value : attr.getValues<float>()) {
            std::cout << value << " ";
        }
    } else if (type.getElementType().isInteger(32)) {
        // For 32-bit integer elements, print each value
        for (int32_t value : attr.getValues<int32_t>()) {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl; // End the line after printing all elements
}

class GraphNode{
    public:
        string name;
        bool is_ts_input = false; // to dicriminate tensornode and Opnode
        vector<GraphNode*> predecessors;
        vector<GraphNode*> successors;//assumption: all nodes have only one output (but can be reused by multiple successors)
    public:
        GraphNode()=default;
        virtual ~GraphNode() = default;
        GraphNode(string name):name(name){};
};

class OPnode: public GraphNode{
    public:
        OpType optype;
        mlir::Operation* mlir_op;
        OPnode(OpType type,string name):GraphNode(name),optype(type){};
};


class TensorNode: public GraphNode{
    public:
        mlir::DenseElementsAttr dense_attr;
        int argument_idx;   // Indicate the index of this tensor in mlir function arguments
        TensorNode(string name, vector<float> value,  mlir::MLIRContext* context):GraphNode(name){
            is_ts_input=true; //mark that it's a tensor nodes
            auto tensorType = mlir::RankedTensorType::get({2, 2}, mlir::FloatType::getF32(context));
            //create a mlir attribute with its value
            dense_attr=mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>(value));         
        }
        void print(){
            print_dense_attr(this->dense_attr);
        }
};

class Graph{
    public:
        Graph():num_ops(0){};
        void push_op(OPnode* node){
            //push operation with no input
            op_nodes.push_back(node);
            num_ops ++;
        }
        void push_op(OPnode* node,GraphNode* input1){
            //push op with one input
            op_nodes.push_back(node);
            num_ops ++;
            //for unseen tensor inputs, record it and assign a unique index to it
            if(input1->is_ts_input && graph_inputs.find(((TensorNode*)input1)->name)== graph_inputs.end()){
                ((TensorNode*)input1)->argument_idx = graph_inputs.size();
                graph_inputs[((TensorNode*)input1)->name] = (TensorNode*)input1;
            }
            node->predecessors.push_back(input1);
            input1->successors.push_back(node);
        }
        void push_op(OPnode* node,GraphNode* input1, GraphNode* input2){
            //push op with two inputs
            op_nodes.push_back(node);
            num_ops ++;
            //for unseen tensor inputs, record it and assign a unique index to it
            if(input1->is_ts_input && graph_inputs.find(((TensorNode*)input1)->name)== graph_inputs.end() ){
                ((TensorNode*)input1)->argument_idx = graph_inputs.size();
                graph_inputs[((TensorNode*)input1)->name] = (TensorNode*)input1;
            }
            if(input2->is_ts_input  && graph_inputs.find(((TensorNode*)input2)->name)== graph_inputs.end()){
                ((TensorNode*)input2)->argument_idx = graph_inputs.size();
                graph_inputs[((TensorNode*)input2)->name] = (TensorNode*)input2;
            }
            node->predecessors.push_back(input1);
            input1->successors.push_back(node);
            node->predecessors.push_back(input2);
            input2->successors.push_back(node);
        }

        OPnode* get_exit_op(){
            //find and return the exit operation node in the graph
            vector<OPnode*> res;
            for(OPnode* node : op_nodes){
                if(node->successors.empty()) res.push_back(node);
            }
            assert(res.size()==1);
            return res[0];
        }
        void print(){
            //print the graph in string format
            if(num_ops<=0) {
                printf("empty graph.\n");
                return;
            }
            OPnode* exit_node = this->get_exit_op();
            std::cout << to_string(exit_node) <<std::endl;
            printf("\n");
        }
        string to_string(GraphNode* node){
            //helper function for print().
            string res = node->name;
            if(node->predecessors.empty())  return res;
            else{
                res += "(";

                for(GraphNode* pred: node->predecessors)    {
                    res += to_string(pred);
                    res += ", ";
                }
                res += ")";
            }
            return res;
        }
        mlir::Operation* create_operation(mlir::OpBuilder& builder,mlir::Location& loc,llvm::SmallVector<mlir::Value, 2>& arguments,OPnode* node){
            //helper function for _build_mlir()
            switch(node->optype){
                case(OpType::Add):
                    return builder.create<mlir::stablehlo::AddOp>(loc, arguments)
                        .getOperation();
                case(OpType::Subtract):
                    return builder.create<mlir::stablehlo::SubtractOp>(loc, arguments)
                        .getOperation();
                case(OpType::Multiply):
                    return builder.create<mlir::stablehlo::MulOp>(loc, arguments)
                        .getOperation();
                case(OpType::Divide):
                    return builder.create<mlir::stablehlo::DivOp>(loc, arguments)
                        .getOperation();
                
                default:
                    return builder.create<mlir::stablehlo::AddOp>(loc, arguments)
                        .getOperation();
            }
        }
        mlir::Operation* _build_mlir(mlir::OpBuilder& builder,mlir::Location& loc, llvm::SmallVector<mlir::Value, 3>& tesnor_inputs){
            //build the whole graph given entry points. First create entry operations, then create their successors.
            std::queue<GraphNode*> q;
            unordered_set<GraphNode*> visited;
            unordered_map<GraphNode*, int> node_degree;
            for (auto& node: op_nodes) {
                for (auto& node_pred: node->predecessors) {
                    if (!node_pred->is_ts_input) {
                        node_degree[node] ++;
                    }
                }
            }
            for (auto& node: op_nodes) {
                if (node_degree[node] == 0 && !node->is_ts_input) {
                    visited.insert(node);
                    q.push(node);
                }
            }
            while(!q.empty()) {
                auto node = q.front();
                q.pop();

                llvm::SmallVector<mlir::Value, 2> inputs;
                for(GraphNode* pred: node->predecessors){
                    if (pred->is_ts_input) {
                        inputs.push_back(tensor_inputs[((TensorNode*) pred)->argument_idx]);
                    } else {
                        inputs.push_back(((OPnode*)pred)->mlir_op->getResult(0));
                    }
                }

                node->mlir_op = create_operation(builder, loc, inputs, node);

                for(GraphNode* succ: node->successors) {
                    node_degree[succ] --;
                    if (node_degree[succ] == 0 && visited.find(succ) == visited.end()) {
                        q.push(succ);
                        visited.insert(succ);
                    }
                }
            }
            for (auto& node: op_nodes) {
                if (node->successors.empty()) {
                    return node->mlir_op;
                }
            }
            return nullptr;
        }

        void to_mlir(mlir::MLIRContext& context){
            //call _build_mlir() to convert graph to mlir format, then print the mlir module.
            mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
            module->getContext()->loadDialect<mlir::func::FuncDialect>();
            module->getContext()->loadDialect<mlir::stablehlo::StablehloDialect>();
            module->getContext()->loadDialect<mlir::quant::QuantizationDialect>();
            

            /** create function **/
            // create function argument and result types.
            auto tensorType =
                mlir::RankedTensorType::get({2, 2}, mlir::FloatType::getF32(&context));
            auto func_type =
                mlir::FunctionType::get(&context, {tensorType, tensorType,tensorType}, {tensorType});

            // create the function and map arguments.
            llvm::ArrayRef<mlir::NamedAttribute> attrs;
            auto function = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                                        "main", func_type, attrs);
            module->push_back(function);

            // create function block with add operations.
            mlir::Block* block = function.addEntryBlock();
            llvm::SmallVector<mlir::Value, 3> arguments(block->args_begin(),
                                                        block->args_end());
            mlir::OpBuilder block_builder = mlir::OpBuilder::atBlockEnd(block);
            mlir::Location loc = block_builder.getUnknownLoc();

            llvm::SmallVector<mlir::NamedAttribute, 10> attributes;

            mlir::Operation* exit_op =  _build_mlir(block_builder,loc);
            block_builder.create<mlir::func::ReturnOp>(loc, exit_op->getResult(0));

            graph_module = module;

            module->dump();
        }

        llvm::SmallVector<mlir::DenseElementsAttr> execute(mlir::MLIRContext& context){
            llvm::SmallVector<mlir::DenseElementsAttr, 3> input_args;
            for(std::pair<string,TensorNode*> p : this->graph_inputs){
                input_args.push_back(p.second->dense_attr);
            }
            mlir::stablehlo::InterpreterConfiguration config;
            auto results = evalModule(*graph_module, input_args, config);
            return (*results);
        }

        auto exit_node() {
            for (auto& node: op_nodes) {
                if (node->successors.empty()) {
                    return node;
                }
            }
            return nullptr;
        }
    public:
        mlir::OwningOpRef<mlir::ModuleOp>& graph_module;
        vector<OPnode*> op_nodes;
        map<string,TensorNode*> graph_inputs;
        int num_ops;
};

bool find_same_subgraph(GraphNode* gn1, GraphNode* gn2) {
    if (gn1->is_ts_input != gn2->is_ts_input) {
        return false;
    }

    if (gn1->is_ts_input) {
        return ((TensorNode*)gn1)->name == ((TensorNode*)gn2)->name;
    } else {
        if (((OPnode*)gn1)->optype != ((OPnode*)gn2)->optype) {
            return false;
        }
        if (((OPnode*)gn1)->predecessors.size() != ((OPnode*)gn2)->predecessors.size()) {
            return false;
        }
        for (int i = 0; i < ((OPnode*)gn1)->predecessors.size(); i ++) {
            if (!find_same_subgraph(((OPnode*)gn1)->predecessors[i], ((OPnode*)gn2)->predecessors[i])) {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool find_same_supergraph(GraphNode* gn1, GraphNode* gn2) {
    if (gn1->is_ts_input != gn2->is_ts_input) {
        return false;
    }

    int counter = 1; 
    if (gn1->is_ts_input) {
        return ((TensorNode*)gn1)->name == ((TensorNode*)gn2)->name;
    } else {
        for (int i = 0; i < ((OPnode*)gn1)->predecessors.size(); i ++) {
            if (!find_same_subgraph(((OPnode*)gn1)->predecessors[i], ((OPnode*)gn2)->predecessors[i])) {
                counter ++;
            }
        }
        if (counter > 1) return false;
        return true;
    }

    return false;
} 

bool variable_ordering() {
    // TODO:
    return true;
}

bool pass_checks(Graph& g1, Graph& g2) {
    // Check if the two graphs are the same
    for (auto& g1_node: g1.op_nodes) {
        for (auto& g2_node: g2.op_nodes) {
            if (find_same_supergraph(g1_node, g2_node)) {
                return false;
            }
        }
    }
    
    if (find_same_supergraph(g1.exit_node(), g2.exit_node())) {
        return false;
    }

    // variable_ordering();

    return true;
}

//unit test
int main(){
    //define inputs and operations
    mlir::MLIRContext context;
    TensorNode* x = new TensorNode("x",{3,5,7,9},&context);//create a 4-element tensor named x
    TensorNode* y = new TensorNode("y",{1,1,1,1},&context);
    TensorNode* z = new TensorNode("z",{2,4,6,8},&context);
    printf("x: "); x->print();
    printf("y: "); y->print();
    printf("z: "); z->print();
    OPnode* op1 = new OPnode(OpType::Subtract,"subtract");//create a operation op1 named "subtract"
    OPnode* op2 = new OPnode(OpType::Multiply,"mult");
    OPnode* op3 = new OPnode(OpType::Add,"add");
    
    //construct a simple compututation graph
    Graph g1;
    g1.push_op(op1,x,y);//add operation op1, with lhs operand x and rhs operand y 
    g1.push_op(op2,op1,z); //add operation op2, with lhs operand = output of op1, and rhs oprand z
    g1.push_op(op3,y,op2);
    printf("graph: "); g1.print(); 

    //convert graph to mlir 
    g1.to_mlir(context);//create mlir and dump it to stderr
    
    //execute converted mlir
    auto res = g1.execute(context);
    printf("output: "); print_dense_attr(res[0]);

    return 0;
}
