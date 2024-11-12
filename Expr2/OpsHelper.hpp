enum class OpResultType { Scalar, Vector };

// 操作符描述
struct OperatorDescriptor {
    int minArgs;
    int maxArgs;
    int deltaStack;
    OpResultType resultType;
    std::variant<
        std::function<float(const std::vector<float>&)>,
        std::function<std::vector<float>(const std::vector<float>&)>
    > func;
    
    static OperatorDescriptor createUnary(std::function<float(float)> f) {
        return {
            1, 1, 0, 
            OpResultType::Scalar,
            [f](const std::vector<float>& args) { return f(args[0]); }
        };
    }
    
    static OperatorDescriptor createBinary(std::function<float(float,float)> f) {
        return {
            2, 2, -1, 
            OpResultType::Scalar,
            [f](const std::vector<float>& args) { return f(args[0], args[1]); }
        };
    }
    
    static OperatorDescriptor createTernary(std::function<float(float,float,float)> f) {
        return {
            3, 3, -2, 
            OpResultType::Scalar,
            [f](const std::vector<float>& args) { return f(args[0], args[1], args[2]); }
        };
    }
    
    static OperatorDescriptor createStackOp(int minA, int maxA, int delta, 
        std::function<std::vector<float>(const std::vector<float>&)> f) 
    {
        return {minA, maxA, delta, OpResultType::Vector, f};
    }
};

struct OperatorPattern {
    std::string prefix;
    std::function<bool(const std::string&)> matcher;
    std::function<OperatorDescriptor(const std::string&)> generator;
};

#define DEFINE_UNARY_OP(name, impl) \
    {name, {1, 1, 0, OpResultType::Scalar, \
        [](const std::vector<float>& args) -> float { return impl(args[0]); }}}

#define DEFINE_BINARY_OP(name, impl) \
    {name, {2, 2, -1, OpResultType::Scalar, \
        [](const std::vector<float>& args) -> float { return impl(args[0], args[1]); }}}

#define DEFINE_TERNARY_OP(name, impl) \
    {name, {3, 3, -2, OpResultType::Scalar, \
        [](const std::vector<float>& args) -> float { return impl(args[0], args[1], args[2]); }}}

#define DEFINE_SWAP_OP(start) \
    {"swap" #start, {(start + 1), (start + 1), 0, OpResultType::Vector, \
        [start](const std::vector<float>& args) -> std::vector<float> { \
            std::vector<float> results = args; \
            std::swap(results[0], results[start]); \
            return results; \
        }}}

#define DEFINE_DUP_OP(start) \
    {"dup" #start, {(start + 1), (start + 1), 1, OpResultType::Vector, \
        [start](const std::vector<float>& args) -> std::vector<float> { \
            std::vector<float> results = args; \
            results.push_back(args[args.size() - start - 1]); \
            return results; \
        }}}

// 通用基础模板：用于递归终止，当 start == end 时停止递归
template<int start, int end, void(*InsertFunc)(int, std::map<std::string, OperatorDescriptor>&)>
struct OpsRange {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        InsertFunc(start, opMap);  // 使用传入的 InsertFunc 函数进行插入
        OpsRange<start + 1, end, InsertFunc>::apply(opMap);  // 递归调用，展开下一个索引
    }
};

// 通用特化模板：用于递归终止
template<int end, void(*InsertFunc)(int, std::map<std::string, OperatorDescriptor>&)>
struct OpsRange<end, end, InsertFunc> {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        InsertFunc(end, opMap);  // 最后一次展开并插入
    }
};

// Swap 操作符插入函数
void insertSwapOp(int start, std::map<std::string, OperatorDescriptor>& opMap) {
    opMap.insert(DEFINE_SWAP_OP(start));  // 插入 swap 操作符
}

// Dup 操作符插入函数
void insertDupOp(int start, std::map<std::string, OperatorDescriptor>& opMap) {
    opMap.insert(DEFINE_DUP_OP(start));  // 插入 dup 操作符
}

// 使用 OpsRange 模板来定义 SwapOpsRange 和 DupOpsRange
template<int start, int end>
using SwapOpsRange = OpsRange<start, end, insertSwapOp>;

template<int start, int end>
using DupOpsRange = OpsRange<start, end, insertDupOp>;