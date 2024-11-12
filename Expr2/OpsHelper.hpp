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

#define DEFINE_DUP_OP(n) \
    {"dup" #n, {(n + 1), (n + 1), 1, OpResultType::Vector, \
        [](const std::vector<float>& args) -> std::vector<float> { \
            std::vector<float> results = args; \
            results.push_back(args[args.size() - n - 1]); \
            return results; \
        }}}

#define DEFINE_SWAP_OP(n) \
    {"swap" #n, {(n + 1), (n + 1), 0, OpResultType::Vector, \
        [](const std::vector<float>& args) -> std::vector<float> { \
            std::vector<float> results = args; \
            std::swap(results[0], results[n]); \
            return results; \
        }}}

// 基础模板：用于递归终止，当 start == end 时停止递归
template<int start, int end>
struct SwapOpsRange {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_SWAP_OP(start));  // 执行当前索引的宏插入
        SwapOpsRange<start + 1, end>::apply(opMap);  // 递归调用，展开下一个索引
    }
};

// 特化模板：用于递归终止
template<int end>
struct SwapOpsRange<end, end> {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_SWAP_OP(end));  // 最后一次展开并插入
    }
};

// 基础模板：用于递归终止，当 start == end 时停止递归
template<int start, int end>
struct DupOpsRange {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_DUP_OP(start));  // 执行当前索引的宏插入
        DupOpsRange<start + 1, end>::apply(opMap);  // 递归调用，展开下一个索引
    }
};

// 特化模板：用于递归终止
template<int end>
struct DupOpsRange<end, end> {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_DUP_OP(end));  // 最后一次展开并插入
    }
};

// 允许操作符表为 const 的版本
template<int start, int end>
struct SwapOpsRangeConst {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_SWAP_OP(start));  // 执行当前索引的宏插入
        SwapOpsRangeConst<start + 1, end>::apply(opMap);  // 递归调用，展开下一个索引
    }
};

template<int end>
struct SwapOpsRangeConst<end, end> {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_SWAP_OP(end));  // 最后一次展开并插入
    }
};

template<int start, int end>
struct DupOpsRangeConst {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_DUP_OP(start));  // 执行当前索引的宏插入
        DupOpsRangeConst<start + 1, end>::apply(opMap);  // 递归调用，展开下一个索引
    }
};

template<int end>
struct DupOpsRangeConst<end, end> {
    static void apply(std::map<std::string, OperatorDescriptor>& opMap) {
        opMap.insert(DEFINE_DUP_OP(end));  // 最后一次展开并插入
    }
};