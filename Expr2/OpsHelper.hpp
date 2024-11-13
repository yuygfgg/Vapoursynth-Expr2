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

// 编译期整数序列生成
template<std::size_t... Is>
struct index_sequence {};

template<std::size_t N, std::size_t... Is>
struct make_index_sequence_impl : make_index_sequence_impl<N-1, N-1, Is...> {};

template<std::size_t... Is>
struct make_index_sequence_impl<0, Is...> : index_sequence<Is...> {};

template<std::size_t N>
using make_index_sequence = make_index_sequence_impl<N>;

// 操作符生成器基类
template<typename Derived>
struct OpGeneratorBase {
    template<size_t I>
    static auto makeOp() {
        return Derived::template generate<I>();
    }

    template<typename Map, size_t... Is>
    static void generateOps(Map& map, index_sequence<Is...>) {
        (map.insert(makeOp<Is>()), ...);
    }
};

// Dup操作符生成器
struct DupOpGenerator : OpGeneratorBase<DupOpGenerator> {
    template<size_t I>
    static auto generate() {
        return std::make_pair(
            std::string("dup") + std::to_string(I),
            OperatorDescriptor{
                static_cast<int>(I + 1), static_cast<int>(I + 1), 1,
                OpResultType::Vector,
                [](const std::vector<float>& args) -> std::vector<float> {
                    std::vector<float> results = args;
                    results.push_back(args[args.size() - I - 1]);
                    return results;
                }
            }
        );
    }
};

// Swap操作符生成器
struct SwapOpGenerator : OpGeneratorBase<SwapOpGenerator> {
    template<size_t I>
    static auto generate() {
        return std::make_pair(
            std::string("swap") + std::to_string(I),
            OperatorDescriptor{
                static_cast<int>(I + 1), static_cast<int>(I + 1), 0,
                OpResultType::Vector,
                [](const std::vector<float>& args) -> std::vector<float> {
                    std::vector<float> results = args;
                    std::swap(results[0], results[I]);
                    return results;
                }
            }
        );
    }
};