#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>
#include <stack>
#include <map>
#include <set>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <memory>
#include <algorithm>
#include <iostream>

#include "OpsHelper.hpp"

const int MAX_STACK_SIZE = 4096; // 4096 should be long enough

// 异常类
class ExprError : public std::exception {
private:
    std::string msg;
public:
    ExprError(const std::string& message) : msg(message) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

class OperatorRegistry {
private:

    static std::map<std::string_view, OperatorDescriptor>& getStaticOperators() {
        static std::map<std::string_view, OperatorDescriptor> staticOperators = {
            // 基本算术运算符
            DEFINE_BINARY_OP("+", [](float a, float b) { return a + b; }),
            DEFINE_BINARY_OP("-", [](float a, float b) { return a - b; }),
            DEFINE_BINARY_OP("*", [](float a, float b) { return a * b; }),
            DEFINE_BINARY_OP("/", [](float a, float b) { 
                [[unlikely]] if (b == 0) throw ExprError("Division by zero");
                return a / b; 
            }),

            // 数学函数
            DEFINE_UNARY_OP("exp", [](float a) {
                [[unlikely]] if (a > 88) throw ExprError("exp overflow");
                return std::exp(a);
            }),
            DEFINE_UNARY_OP("log", [](float a) {
                [[unlikely]] if (a <= 0) throw ExprError("log domain error");
                return std::log(a);
            }),
            DEFINE_UNARY_OP("sqrt", [](float a) {
                [[unlikely]] if (a < 0) throw ExprError("sqrt domain error");
                return std::sqrt(a);
            }),
            DEFINE_UNARY_OP("sin", std::sin),
            DEFINE_UNARY_OP("cos", std::cos),
            DEFINE_UNARY_OP("abs", std::abs),
            DEFINE_UNARY_OP("not", [](float a) { return a > 0 ? 0.0f : 1.0f; }),

            // 比较运算符
            DEFINE_BINARY_OP(">", [](float a, float b) { return a > b ? 1.0f : 0.0f; }),
            DEFINE_BINARY_OP("<", [](float a, float b) { return a < b ? 1.0f : 0.0f; }),
            DEFINE_BINARY_OP("=", [](float a, float b) { return a == b ? 1.0f : 0.0f; }),
            DEFINE_BINARY_OP(">=", [](float a, float b) { return a >= b ? 1.0f : 0.0f; }),
            DEFINE_BINARY_OP("<=", [](float a, float b) { return a <= b ? 1.0f : 0.0f; }),

            // 逻辑运算符
            DEFINE_BINARY_OP("and", [](float a, float b) { return (a > 0 && b > 0) ? 1.0f : 0.0f; }),
            DEFINE_BINARY_OP("or", [](float a, float b) { return (a > 0 || b > 0) ? 1.0f : 0.0f; }),
            DEFINE_BINARY_OP("xor", [](float a, float b) { return ((a > 0) != (b > 0)) ? 1.0f : 0.0f; }),

            // 其他数学运算符
            DEFINE_BINARY_OP("max", std::max),
            DEFINE_BINARY_OP("min", std::min),
            DEFINE_BINARY_OP("pow", [](float a, float b) { 
                [[unlikely]] if (a < 0) throw ExprError("pow domain error");
                float result = std::pow(a, b);
                [[unlikely]] if (result > 3e38f || result < 1e-38f) 
                    throw ExprError("pow range error");
                return result;
            }),

            // 三元运算符
            DEFINE_TERNARY_OP("?", [](float c, float t, float f) { return c > 0 ? t : f; }),

            {
                "dup", {1, 1, 1, OpResultType::Vector,
                    [](const std::vector<float>& args) -> std::vector<float> {
                        std::vector<float> results = args;
                        results.push_back(args.back());
                        return results;
                    }
                }
            },
            {
                "swap", {2, 2, 0, OpResultType::Vector,
                    [](const std::vector<float>& args) -> std::vector<float> {
                        std::vector<float> results = args;
                        std::swap(results[0], results[1]);
                        return results;
                    }
                }
            }
        };

        // 在编译期生成 dup1 到 dup25
        DupOpGenerator::generateOps(staticOperators, make_index_sequence<25>{});
        
        // 在编译期生成 swap1 到 swap25
        SwapOpGenerator::generateOps(staticOperators, make_index_sequence<25>{});

        return staticOperators;
    }

public:
    const OperatorDescriptor* getOperator(const std::string& token) const {
        const auto& staticOps = getStaticOperators();
        auto staticIt = staticOps.find(token);
        if (staticIt != staticOps.end()) {
            return &staticIt->second;
        }
        return nullptr;
    }
};

// 栈操作类
class ExprStack {
private:
    std::stack<float> stack;
    
    float pop() {
        [[unlikely]] if (stack.empty()) throw ExprError("Stack underflow");
        float val = stack.top();
        stack.pop();
        return val;
    }

public:
    void push(float value) { stack.push(value); }
    
    // 收集操作数
    std::vector<float> collectArgs(size_t n) {
        [[unlikely]] if (stack.size() < n) {
            throw ExprError("Stack underflow: need " + std::to_string(n) + " values");
        }
        std::vector<float> args(n);
        for (size_t i = 0; i < n; ++i) {
            args[n - 1 - i] = pop();
        }
        return args;
    }

    // 应用结果
    void applyResults(const std::vector<float>& results) {
        for (float value : results) {
            push(value);
        }
    }

    float getResult() const {
        [[unlikely]] if (stack.size() != 1) {
            throw ExprError("Invalid expression evaluation: stack size = " + 
                            std::to_string(stack.size()));
        }
        return stack.top();
    }

    void clear() { while (!stack.empty()) stack.pop(); }
    size_t size() const { return stack.size(); }
    
    std::string debugStack() const {
        std::string result = "Stack (bottom to top): ";
        std::vector<float> values;
        auto temp = stack;
        while (!temp.empty()) {
            values.push_back(temp.top());
            temp.pop();
        }
        std::reverse(values.begin(), values.end());
        for (float v : values) {
            result += std::to_string(v) + " ";
        }
        return result;
    }
};

// 语法检查器
class ExpressionValidator {
private:
    const OperatorRegistry& registry;
    
    static bool isNumber(const std::string& token) {
        try {
            size_t pos;
            std::stof(token, &pos);
            return pos == token.length();
        } catch (...) {
            return false;
        }
    }

    static bool isVariable(const std::string& token) {
        if (token == "x" || token == "y" || token == "z") return true;
        return token.length() == 1 && token[0] >= 'a' && token[0] <= 'w';
    }

    void validateStackEffect(const std::vector<std::string>& tokens) const {
        int stackSize = 0;
        int maxStackSize = 0;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            
            [[likely]] if (isNumber(token) || isVariable(token)) {
                stackSize++;
            }
            else if (auto op = registry.getOperator(token)) {
                if (stackSize < op->minArgs) {
                    throw ExprError("Stack underflow at position " + std::to_string(i) + 
                                    ": operator '" + token + "' requires " + 
                                    std::to_string(op->minArgs) + " operands");
                }
                stackSize += op->deltaStack;
            }
            else {
                throw ExprError("Invalid token at position " + std::to_string(i) + 
                                ": '" + token + "'");
            }
            
            maxStackSize = std::max(maxStackSize, stackSize);
            [[unlikely]] if (maxStackSize > MAX_STACK_SIZE) {
                throw ExprError("Stack size limit exceeded");
            }
        }
        
        [[unlikely]] if (stackSize != 1) {
            throw ExprError("Invalid expression: should leave exactly one value on stack, " + 
                            std::to_string(stackSize) + " values found");
        }
    }

public:
    ExpressionValidator(const OperatorRegistry& reg) : registry(reg) {}

    void validate(const std::string& expr, int numClips) const {
        if (expr.empty()) return;  // 空表达式是合法的

        std::vector<std::string> tokens;
        std::stringstream ss(expr);
        std::string token;
        
        while (ss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) return;

        // 检查变量引用是否在有效范围内
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            if (token == "x") {
                [[unlikely]] if (numClips < 1) throw ExprError(
                    "Variable 'x' used but no clips provided");
            }
            else if (token == "y") {
                [[unlikely]] if (numClips < 2) throw ExprError(
                    "Variable 'y' used but only " + std::to_string(numClips) + " clip(s) provided");
            }
            else if (token == "z") {
                [[unlikely]] if (numClips < 3) throw ExprError(
                    "Variable 'z' used but only " + std::to_string(numClips) + " clip(s) provided");
            }
            else if (token.length() == 1 && token[0] >= 'a' && token[0] <= 'w') {
                int required = token[0] - 'a' + 4;
                [[unlikely]] if (numClips < required) throw ExprError(
                    "Variable '" + token + "' used but only " + 
                    std::to_string(numClips) + " clip(s) provided");
            }
        }

        // 检查栈操作的有效性
        validateStackEffect(tokens);
    }
};

struct ExecutableToken {
    enum class Type {
        Operator,
        Variable,
        Constant
    } type;
    
    union {
        const OperatorDescriptor* op;
        int varIndex;
        float constant;
    };

    ExecutableToken(const OperatorDescriptor* op) : type(Type::Operator), op(op) {}
    ExecutableToken(int var) : type(Type::Variable), varIndex(var) {}
    ExecutableToken(float val) : type(Type::Constant), constant(val) {}
};


// 表达式计算器
class ExprCalculator {
private:
    OperatorRegistry registry;
    ExpressionValidator validator;

    std::vector<std::string> tokenize(const std::string& expr) const {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(expr);
        while (std::getline(tokenStream, token, ' ')) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    std::vector<ExecutableToken> compileExpression(const std::string& expr) const {
        std::vector<ExecutableToken> result;
        auto tokens = tokenize(expr);
        
        for (const auto& token : tokens) {
            if (auto op = registry.getOperator(token)) {
                result.emplace_back(op);
            }
            else if (token == "x") {
                result.emplace_back(0);
            }
            else if (token == "y") {
                result.emplace_back(1);
            }
            else if (token == "z") {
                result.emplace_back(2);
            }
            else if (token.length() == 1 && token[0] >= 'a' && token[0] <= 'w') {
                result.emplace_back(token[0] - 'a' + 3);
            }
            else {
                result.emplace_back(std::stof(token));
            }
        }
        
        return result;
    }

    float evaluatePixel(const std::vector<ExecutableToken>& compiled, 
                        const std::vector<float>& values) const {
        ExprStack stack;

        for (const auto& token : compiled) {
            try {
                switch (token.type) {
                    case ExecutableToken::Type::Operator: {
                        const auto& op = token.op;
                        auto args = stack.collectArgs(op->minArgs);

                        if (op->resultType == OpResultType::Scalar) {
                            auto f = std::get<std::function<float(const std::vector<float>&)>>(op->func);
                            stack.push(f(args));
                        } else {
                            auto f = std::get<std::function<std::vector<float>(const std::vector<float>&)>>(op->func);
                            auto results = f(args);
                            stack.applyResults(results);
                        }
                        break;
                    }
                    case ExecutableToken::Type::Variable: {
                        int index = token.varIndex;
                        stack.push(values.size() > index ? values[index] : 0.0f);
                        break;
                    }
                    case ExecutableToken::Type::Constant: {
                        stack.push(token.constant);
                        break;
                    }
                }
            }
            catch (const std::exception& e) {
                throw ExprError(std::string("Evaluation error: ") + e.what() + ". " + stack.debugStack());
            }
        }

        return stack.getResult();
    }

public:
    ExprCalculator() : validator(registry) {}

    void validate(const std::string& expr, int numClips) const {
        validator.validate(expr, numClips);
    }

    void processPlane(const std::string& expr,
                    const std::vector<const uint8_t*>& srcps,
                    const std::vector<int>& src_strides,
                    uint8_t* dstp,
                    int dst_stride,
                    int width,
                    int height,
                    int plane,
                    const VSFormat* fi) const 
    {
        bool isFloat = fi->sampleType == stFloat;
        int bits = fi->bitsPerSample;
        int maxValue = isFloat ? 1 : ((1 << bits) - 1);
        bool isChroma = (plane == 1 || plane == 2) && 
                        (fi->colorFamily == cmYUV || fi->colorFamily == cmYCoCg);

        auto compiled = compileExpression(expr);
        std::vector<float> pixel_values(srcps.size());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (size_t i = 0; i < srcps.size(); i++) {
                    float value;
                    if (isFloat) {
                        value = reinterpret_cast<const float*>(srcps[i] + y * src_strides[i])[x];
                        if (isChroma) {
                            value += 0.5f;
                        }
                    } else {
                        if (bits > 8) {
                            value = static_cast<float>(reinterpret_cast<const uint16_t*>(
                                srcps[i] + y * src_strides[i])[x]);
                        } else {
                            value = static_cast<float>(srcps[i][y * src_strides[i] + x]);
                        }
                    }
                    pixel_values[i] = value;
                }

                float result = evaluatePixel(compiled, pixel_values);

                if (isFloat) {
                    if (isChroma) {
                        result -= 0.5f;
                    }
                    reinterpret_cast<float*>(dstp + y * dst_stride)[x] = result;
                } else {
                    result = std::clamp(result, 0.0f, static_cast<float>(maxValue));
                    if (bits > 8) {
                        reinterpret_cast<uint16_t*>(dstp + y * dst_stride)[x] =
                            static_cast<uint16_t>(result);
                    } else {
                        dstp[y * dst_stride + x] = static_cast<uint8_t>(result);
                    }
                }
            }
        }
    }
};


// VapourSynth 插件结构
struct ExprData {
    VSNodeRef* nodes[26];  // 支持x,y,z + a-w共26个输入clip
    int numNodes;
    const VSVideoInfo* vi;
    std::vector<std::string> expressions;  // 每个平面一个表达式
    ExprCalculator calculator;
};

// 检查所有输入clip的格式是否兼容
static bool checkVideoFormats(const VSVideoInfo* vi1, const VSVideoInfo* vi2) {
    return  vi1->format->colorFamily == vi2->format->colorFamily &&
            vi1->format->sampleType == vi2->format->sampleType &&
            vi1->format->bitsPerSample == vi2->format->bitsPerSample &&
            vi1->format->subSamplingW == vi2->format->subSamplingW &&
            vi1->format->subSamplingH == vi2->format->subSamplingH &&
            vi1->width == vi2->width &&
            vi1->height == vi2->height;
}

static void VS_CC exprInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi) {
    ExprData* d = static_cast<ExprData*>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef* VS_CC exprGetFrame(int n, int activationReason, void** instanceData, void** frameData,
    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    ExprData* d = static_cast<ExprData*>(*instanceData);

    if (activationReason == arInitial) {
        for (int i = 0; i < d->numNodes; i++) {
            vsapi->requestFrameFilter(n, d->nodes[i], frameCtx);
        }
        return nullptr;
    }

    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    VSFrameRef* dst = nullptr;
    std::vector<const VSFrameRef*> src_frames(d->numNodes, nullptr);

    try {
        // 获取所有输入帧
        for (int i = 0; i < d->numNodes; i++) {
            src_frames[i] = vsapi->getFrameFilter(n, d->nodes[i], frameCtx);
        }

        // 创建输出帧
        const VSFrameRef* first_frame = src_frames[0];
        dst = vsapi->newVideoFrame(d->vi->format,
            vsapi->getFrameWidth(first_frame, 0),
            vsapi->getFrameHeight(first_frame, 0),
            first_frame, core);

        const VSFormat* fi = d->vi->format;
        int planes = fi->numPlanes;

        for (int plane = 0; plane < planes; plane++) {
            if (d->expressions[plane].empty()) {
                // 空表达式时直接复制
                const uint8_t* srcp = vsapi->getReadPtr(first_frame, plane);
                int src_stride = vsapi->getStride(first_frame, plane);
                uint8_t* dstp = vsapi->getWritePtr(dst, plane);
                int dst_stride = vsapi->getStride(dst, plane);
                int height = vsapi->getFrameHeight(first_frame, plane);
                int rowsize = vsapi->getFrameWidth(first_frame, plane) * fi->bytesPerSample;

                vs_bitblt(dstp, dst_stride, srcp, src_stride, rowsize, height);
                continue;
            }

            // 收集所有源平面的指针和步长
            std::vector<const uint8_t*> srcps;
            std::vector<int> src_strides;
            for (const auto& frame : src_frames) {
                srcps.push_back(vsapi->getReadPtr(frame, plane));
                src_strides.push_back(vsapi->getStride(frame, plane));
            }

            // 处理整个平面
            d->calculator.processPlane(
                d->expressions[plane],
                srcps,
                src_strides,
                vsapi->getWritePtr(dst, plane),
                vsapi->getStride(dst, plane),
                vsapi->getFrameWidth(first_frame, plane),
                vsapi->getFrameHeight(first_frame, plane),
                plane,
                fi
            );
        }

        // 释放源帧
        for (const auto& frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        return dst;
    }
    catch (const std::exception& e) {
        if (dst) vsapi->freeFrame(dst);
        for (const auto& frame : src_frames) {
            if (frame) vsapi->freeFrame(frame);
        }
        vsapi->setFilterError(("Expr: " + std::string(e.what())).c_str(), frameCtx);
        return nullptr;
    }
}

static void VS_CC exprFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    ExprData* d = static_cast<ExprData*>(instanceData);
    for (int i = 0; i < d->numNodes; i++) {
        vsapi->freeNode(d->nodes[i]);
    }
    delete d;
}

static void VS_CC exprCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    try {
        std::unique_ptr<ExprData> d(new ExprData());
        int err;

        // 获取输入clips
        d->numNodes = vsapi->propNumElements(in, "clips");
        [[unlikely]] if (d->numNodes < 1 || d->numNodes > 26) {
            throw ExprError("Must specify between 1 and 26 input clips");
        }

        // 获取所有输入clip并检查格式
        for (int i = 0; i < d->numNodes; i++) {
            d->nodes[i] = vsapi->propGetNode(in, "clips", i, &err);
            const VSVideoInfo* vi = vsapi->getVideoInfo(d->nodes[i]);
            
            if (i == 0) {
                d->vi = vsapi->getVideoInfo(d->nodes[0]);
            } else if (!checkVideoFormats(d->vi, vi)) { [[unlikely]]
                throw ExprError("All inputs must have the same format and dimensions");
            }
        }

        // 获取表达式
        int num_expr = vsapi->propNumElements(in, "expr");
        [[unlikely]] if (num_expr < 1) {
            throw ExprError("At least one expression must be specified");
        }

        // 为每个平面获取表达式并验证
        for (int i = 0; i < num_expr && i < d->vi->format->numPlanes; i++) {
            std::string expr = vsapi->propGetData(in, "expr", i, &err);
            d->calculator.validate(expr, d->numNodes);  // 验证表达式
            d->expressions.push_back(expr);
        }

        // 如果表达式数量少于平面数量，用最后一个表达式补充
        while (d->expressions.size() < d->vi->format->numPlanes) {
            d->expressions.push_back(d->expressions.back());
        }

        vsapi->createFilter(in, out, "Expr", exprInit, exprGetFrame, exprFree, 
            fmParallel, 0, d.release(), core);
    }
    catch (const std::exception& e) {
        vsapi->setError(out, ("Expr: " + std::string(e.what())).c_str());
        return;
    }
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin) {
    configFunc("com.yuygfgg.expr2", "yuygfgg",
        "VapourSynth Expression Evaluation Plugin",
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Expr",
        "clips:clip[];expr:data[];",
        exprCreate, nullptr, plugin);
}

