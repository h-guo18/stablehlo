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

#define MAX_NUM_OPS 8
using namespace std;

enum class OpType {
    Add,
    Subtract,
    Multiply,
    Divide,
    // Add more operation types here
};

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
        OPnode(OpType type,string name):GraphNode(name),optype(type){};
};


class TensorNode: public GraphNode{
    public:
        mlir::DenseElementsAttr value;
        TensorNode(string name):GraphNode(name){
            is_ts_input=true; //mark that it's a tensor nodes
            }
        bool operator==(const TensorNode& node2) const {
            return (value == node2.value);
        }

};

class Graph{
    public:
        Graph():numNodes(0){};
        void push_op(GraphNode* node){
            //push operation with no input
            nodes.push_back(node);
            numNodes ++;
        }
        void push_op(GraphNode* node,GraphNode* input1){
            //push op with one input
            nodes.push_back(node);
            numNodes ++;
            node->predecessors.push_back(input1);
            input1->successors.push_back(node);
        }
        void push_op(GraphNode* node,GraphNode* input1, GraphNode* input2){
            //push op with two inputs
            nodes.push_back(node);
            numNodes ++;
            node->predecessors.push_back(input1);
            input1->successors.push_back(node);
            node->predecessors.push_back(input2);
            input2->successors.push_back(node);
        }

        OPnode* get_exit_op(){
            //find and return the exit operation node in the graph
            vector<GraphNode*> res;
            for(GraphNode* node : nodes){
                if(node->successors.empty()) res.push_back(node);
            }
            assert(res.size()==1);
            return (OPnode*)(res[0]);
        }
        vector<OPnode*> get_entry_ops(){
            //find and return the entry operation in the graph (i.e. OpNodes whose inputs only contains TensorNode, but not OPnode.)
            vector<OPnode*> res;
            for(GraphNode* node : nodes){
                bool is_entry = true;
                for(GraphNode* pred:node->predecessors){
                    if (!pred->is_ts_input){
                        is_entry = false;
                        break;
                    }
                }
                if(is_entry) res.push_back((OPnode*)node);
            }
            return res;
        }
        void print(){
            //print the graph in string format
            if(numNodes<=0) {
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
        mlir::Operation* create_operation(mlir::OpBuilder& builder,mlir::Location& loc,llvm::SmallVector<mlir::Value, 4>& arguments,OPnode* node){
            //helper function for build_graph()
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
        mlir::Operation* build_graph(vector<OPnode*> entrys, mlir::OpBuilder& builder,mlir::Location& loc,llvm::SmallVector<mlir::Value, 4>& arguments,){
            //build the whole graph given entry points. First create entry operations, then create their successors.
            unordered_set<OPnode*> created{};
            for(OPnode* node:entrys){
                mlir::Operation* op = create_operation(block_builder,loc,arguments,node);
            }
        }
        void to_mlir(){
            //call build_graph() to convert graph to mlir format, then print the mlir module.
            mlir::MLIRContext context;
            mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
            module->getContext()->loadDialect<mlir::func::FuncDialect>();
            module->getContext()->loadDialect<mlir::stablehlo::StablehloDialect>();
            module->getContext()->loadDialect<mlir::quant::QuantizationDialect>();
            

            /** create function **/
            // create function argument and result types.
            auto tensorType =
                mlir::RankedTensorType::get({3, 4}, mlir::FloatType::getF32(&context));
            auto func_type =
                mlir::FunctionType::get(&context, {tensorType, tensorType}, {tensorType});

            // create the function and map arguments.
            llvm::ArrayRef<mlir::NamedAttribute> attrs;
            auto function = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                                        "main", func_type, attrs);
            module->push_back(function);

            // create function block with add operations.
            mlir::Block* block = function.addEntryBlock();
            llvm::SmallVector<mlir::Value, 4> arguments(block->args_begin(),
                                                        block->args_end());
            mlir::OpBuilder block_builder = mlir::OpBuilder::atBlockEnd(block);
            mlir::Location loc = block_builder.getUnknownLoc();

            llvm::SmallVector<mlir::NamedAttribute, 10> attributes;
            vector<OPnode*> entrys = get_entry_ops();

            mlir::Operation* exit_op =  build_graph(entrys,block_builder,loc,arguments);
            // block_builder.create<mlir::func::ReturnOp>(loc, op->getResult(0));
            (*module).dump();
            return;
        }

    public:
        vector<GraphNode*> nodes;
        int numNodes;
};


//unit test
int main(){
    TensorNode x1("x1");
    TensorNode x2("x2");
    TensorNode x3("x3");
    OPnode op1(OpType::Subtract,"subtract");
    OPnode op2(OpType::Multiply,"mult");
    OPnode op3(OpType::Divide,"divide");

    Graph g1;
    g1.push_op(&op1,&x1,&x2);
    g1.print();
    g1.push_op(&op2,&op1,&x3); 
    g1.print();
    g1.push_op(&op3,&x1,&op2);
    g1.print();
    for(GraphNode* node:g1.get_entry_ops()) std::cout << node->name <<std::endl;
    g1.to_mlir();
    
    


    return 0;
}
