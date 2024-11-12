#include </usr/local/include/vapoursynth/VapourSynth.h>
#include </usr/local/include/vapoursynth/VSHelper.h>
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

const int MAX_STACK_SIZE = 4096; // 4096 should be long enough

enum class OpResultType { Scalar, Vector };

// 异常类
class ExprError : public std::exception {
private:
    std::string msg;
public:
    ExprError(const std::string& message) : msg(message) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

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
    
    // 修改工厂函数
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

// 操作符注册表
class OperatorRegistry {
private:
    mutable std::map<std::string, OperatorDescriptor> operators;
    std::vector<OperatorPattern> patterns;

    // 辅助函数：创建简单的字符串匹配模式
    static OperatorPattern createSimplePattern(
        const std::string& token, 
        const OperatorDescriptor& desc) 
    {
        return {
            token,
            [token](const std::string& t) { return t == token; },
            [desc](const std::string&) { return desc; }
        };
    }

public:
    void registerPattern(const OperatorPattern& pattern) {
        patterns.push_back(pattern);
    }

    void registerOperator(const std::string& token, const OperatorDescriptor& desc) {
        registerPattern(createSimplePattern(token, desc));
    }

    const OperatorDescriptor* getOperator(const std::string& token) const {
        // 先查找固定操作符
        auto it = operators.find(token);
        if (it != operators.end()) {
            return &it->second;
        }

        // 再匹配模式
        for (const auto& pattern : patterns) {
            if (token.starts_with(pattern.prefix) && pattern.matcher(token)) {
                operators[token] = pattern.generator(token);  // 缓存生成的操作符
                return &operators[token];
            }
        }
        return nullptr;
    }

    void initDefaultOperators() {
        // 基本算术运算符
        registerOperator("+", OperatorDescriptor::createBinary([](float a, float b) { return a + b; }));
        registerOperator("-", OperatorDescriptor::createBinary([](float a, float b) { return a - b; }));
        registerOperator("*", OperatorDescriptor::createBinary([](float a, float b) { return a * b; }));
        registerOperator("/", OperatorDescriptor::createBinary([](float a, float b) { 
            if (b == 0) throw ExprError("Division by zero");
            return a / b; 
        }));

        // 数学函数
        registerOperator("exp", OperatorDescriptor::createUnary([](float a) {
            if (a > 88) throw ExprError("exp overflow");
            return std::exp(a);
        }));
        registerOperator("log", OperatorDescriptor::createUnary([](float a) {
            if (a <= 0) throw ExprError("log domain error");
            return std::log(a);
        }));
        registerOperator("sqrt", OperatorDescriptor::createUnary([](float a) {
            if (a < 0) throw ExprError("sqrt domain error");
            return std::sqrt(a);
        }));
        registerOperator("sin", OperatorDescriptor::createUnary([](float a) { return std::sin(a); }));
        registerOperator("cos", OperatorDescriptor::createUnary([](float a) { return std::cos(a); }));
        registerOperator("abs", OperatorDescriptor::createUnary([](float a) { return std::abs(a); }));
        registerOperator("not", OperatorDescriptor::createUnary([](float a) { return a > 0 ? 0.0f : 1.0f; }));

        // 比较运算符
        registerOperator(">", OperatorDescriptor::createBinary([](float a, float b) { return a > b ? 1.0f : 0.0f; }));
        registerOperator("<", OperatorDescriptor::createBinary([](float a, float b) { return a < b ? 1.0f : 0.0f; }));
        registerOperator("=", OperatorDescriptor::createBinary([](float a, float b) { return a == b ? 1.0f : 0.0f; }));
        registerOperator(">=", OperatorDescriptor::createBinary([](float a, float b) { return a >= b ? 1.0f : 0.0f; }));
        registerOperator("<=", OperatorDescriptor::createBinary([](float a, float b) { return a <= b ? 1.0f : 0.0f; }));

        // 逻辑运算符
        registerOperator("and", OperatorDescriptor::createBinary([](float a, float b) { 
            return (a > 0 && b > 0) ? 1.0f : 0.0f; 
        }));
        registerOperator("or", OperatorDescriptor::createBinary([](float a, float b) { 
            return (a > 0 || b > 0) ? 1.0f : 0.0f; 
        }));
        registerOperator("xor", OperatorDescriptor::createBinary([](float a, float b) { 
            return ((a > 0) != (b > 0)) ? 1.0f : 0.0f; 
        }));

        // 其他数学运算符
        registerOperator("max", OperatorDescriptor::createBinary([](float a, float b) { return std::max(a, b); }));
        registerOperator("min", OperatorDescriptor::createBinary([](float a, float b) { return std::min(a, b); }));
        registerOperator("pow", OperatorDescriptor::createBinary([](float a, float b) { 
            if (a < 0) throw ExprError("pow domain error");
            float result = std::pow(a, b);
            if (result > 3e38f || result < 1e-38f) 
                throw ExprError("pow range error");
            return result;
        }));

        // 三元运算符
        registerOperator("?", OperatorDescriptor::createTernary([](float c, float t, float f) { 
            return c > 0 ? t : f; 
        }));

        // 特殊模式：dupN
        registerPattern({
            "dup",
            [](const std::string& token) {
                if (token == "dup") return true;
                if (!token.starts_with("dup")) return false;
                try {
                    int n = std::stoi(token.substr(3));
                    return n >= 0 && n <= 25;
                } catch (...) {
                    return false;
                }
            },
            [](const std::string& token) {
                int n = token == "dup" ? 0 : std::stoi(token.substr(3));
                return OperatorDescriptor::createStackOp(
                    n + 1, n + 1, 1,
                    [n](const std::vector<float>& args) {
                        std::vector<float> results = args;
                        results.push_back(args[args.size() - n - 1]);
                        return results;
                    }
                );
            }
        });

        // 特殊模式：swapN
        registerPattern({
            "swap",
            [](const std::string& token) {
                if (token == "swap") return true;
                if (!token.starts_with("swap")) return false;
                try {
                    int n = std::stoi(token.substr(4));
                    return n >= 1 && n <= 25;
                } catch (...) {
                    return false;
                }
            },
            [](const std::string& token) {
                int n = token == "swap" ? 1 : std::stoi(token.substr(4));
                return OperatorDescriptor::createStackOp(
                    n + 1, n + 1, 0,
                    [n](const std::vector<float>& args) {
                        std::vector<float> results = args;
                        std::swap(results[0], results[n]);
                        return results;
                    }
                );
            }
        });
    }
};

// 栈操作类
class ExprStack {
private:
    std::stack<float> stack;
    
    float pop() {
        if (stack.empty()) throw ExprError("Stack underflow");
        float val = stack.top();
        stack.pop();
        return val;
    }

public:
    void push(float value) { stack.push(value); }
    
    // 收集操作数
    std::vector<float> collectArgs(size_t n) {
        if (stack.size() < n) {
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
        if (stack.size() != 1) {
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

    void validateStackEffect(const std::vector<std::string>& tokens) {
        int stackSize = 0;
        int maxStackSize = 0;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            
            if (isNumber(token) || isVariable(token)) {
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
            if (maxStackSize > MAX_STACK_SIZE) {
                throw ExprError("Stack size limit exceeded");
            }
        }
        
        if (stackSize != 1) {
            throw ExprError("Invalid expression: should leave exactly one value on stack, " + 
                            std::to_string(stackSize) + " values found");
        }
    }

public:
    ExpressionValidator(const OperatorRegistry& reg) : registry(reg) {}

    void validate(const std::string& expr, int numClips) {
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
                if (numClips < 1) throw ExprError(
                    "Variable 'x' used but no clips provided");
            }
            else if (token == "y") {
                if (numClips < 2) throw ExprError(
                    "Variable 'y' used but only " + std::to_string(numClips) + " clip(s) provided");
            }
            else if (token == "z") {
                if (numClips < 3) throw ExprError(
                    "Variable 'z' used but only " + std::to_string(numClips) + " clip(s) provided");
            }
            else if (token.length() == 1 && token[0] >= 'a' && token[0] <= 'w') {
                int required = token[0] - 'a' + 4;
                if (numClips < required) throw ExprError(
                    "Variable '" + token + "' used but only " + 
                    std::to_string(numClips) + " clip(s) provided");
            }
        }

        // 检查栈操作的有效性
        validateStackEffect(tokens);
    }
};

// 表达式计算器
class ExprCalculator {
private:
    OperatorRegistry registry;
    ExprStack stack;
    ExpressionValidator validator;

    std::vector<std::string> tokenize(const std::string& expr) {
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

public:
    ExprCalculator() : validator(registry) {
        registry.initDefaultOperators();
    }

    void validate(const std::string& expr, int numClips) {
        validator.validate(expr, numClips);
    }

    float evaluate(const std::string& expr, const std::vector<float>& values) {
        if (expr.empty()) {
            return values.empty() ? 0.0f : values[0];
        }

        try {
            stack.clear();
            auto tokens = tokenize(expr);
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto& token = tokens[i];
                try {
                    if (auto op = registry.getOperator(token)) {
                        auto args = stack.collectArgs(op->minArgs);
                        if (op->resultType == OpResultType::Scalar) {
                            auto f = std::get<std::function<float(const std::vector<float>&)>>(op->func);
                            stack.push(f(args));
                        } else {
                            auto f = std::get<std::function<std::vector<float>(const std::vector<float>&)>>(op->func);
                            stack.applyResults(f(args));
                        }
                    }
                    else if (token == "x") {
                        stack.push(values.size() > 0 ? values[0] : 0.0f);
                    }
                    else if (token == "y") {
                        stack.push(values.size() > 1 ? values[1] : 0.0f);
                    }
                    else if (token == "z") {
                        stack.push(values.size() > 2 ? values[2] : 0.0f);
                    }
                    else if (token.length() == 1 && token[0] >= 'a' && token[0] <= 'w') {
                        int index = token[0] - 'a' + 3;
                        stack.push(values.size() > index ? values[index] : 0.0f);
                    }
                    else {
                        stack.push(std::stof(token));
                    }
                }
                catch (const std::exception& e) {
                    throw ExprError("Error at token '" + token + "' (position " + 
                                std::to_string(i) + "): " + e.what() + 
                                "\n" + stack.debugStack());
                }
            }

            return stack.getResult();
        }
        catch (const ExprError& e) {
            throw;
        }
        catch (const std::exception& e) {
            throw ExprError("Expression evaluation error: " + std::string(e.what()));
        }
    }
    void registerOperator(const std::string& name, const OperatorDescriptor& desc) {
        registry.registerOperator(name, desc);
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

        // 处理每个平面
        const VSFormat* fi = d->vi->format;
        int planes = fi->numPlanes;
        int bits = fi->bitsPerSample;
        bool isFloat = fi->sampleType == stFloat;
        int maxValue = isFloat ? 1 : ((1 << bits) - 1);

        std::vector<float> pixel_values(d->numNodes);

        for (int plane = 0; plane < planes; plane++) {
            if (d->expressions[plane].empty()) {
                // 空表达式时直接复制第一个clip的对应平面
                const uint8_t* srcp = vsapi->getReadPtr(first_frame, plane);
                int src_stride = vsapi->getStride(first_frame, plane);
                uint8_t* dstp = vsapi->getWritePtr(dst, plane);
                int dst_stride = vsapi->getStride(dst, plane);
                int height = vsapi->getFrameHeight(first_frame, plane);
                int rowsize = vsapi->getFrameWidth(first_frame, plane) * fi->bytesPerSample;

                vs_bitblt(dstp, dst_stride, srcp, src_stride, rowsize, height);
                continue;
            }

            int height = vsapi->getFrameHeight(first_frame, plane);
            int width = vsapi->getFrameWidth(first_frame, plane);
            uint8_t* dstp = vsapi->getWritePtr(dst, plane);
            int dst_stride = vsapi->getStride(dst, plane);

            // 处理每个像素
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // 收集所有输入clip在当前位置的像素值
                    for (int i = 0; i < d->numNodes; i++) {
                        const uint8_t* srcp = vsapi->getReadPtr(src_frames[i], plane);
                        int src_stride = vsapi->getStride(src_frames[i], plane);

                        float value;
                        if (isFloat) {
                            value = reinterpret_cast<const float*>(srcp + y * src_stride)[x];
                            if ((plane == 1 || plane == 2) &&
                                (fi->colorFamily == cmYUV || fi->colorFamily == cmYCoCg)) {
                                value += 0.5f;
                            }
                        } else {
                            if (bits > 8) {
                                value = static_cast<float>(reinterpret_cast<const uint16_t*>(
                                    srcp + y * src_stride)[x]);
                            } else {
                                value = static_cast<float>(srcp[y * src_stride + x]);
                            }
                        }
                        pixel_values[i] = value;
                    }

                    // 计算表达式
                    float result = d->calculator.evaluate(d->expressions[plane], pixel_values);

                    // 写入结果
                    if (isFloat) {
                        if ((plane == 1 || plane == 2) &&
                            (fi->colorFamily == cmYUV || fi->colorFamily == cmYCoCg)) {
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

        // 处理完毕后，释放输入帧
        for (const auto& frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        return dst;
    }
    catch (const std::exception& e) {
        if (dst) {
            vsapi->freeFrame(dst);
        }
        for (const auto& frame : src_frames) {
            if (frame) {
                vsapi->freeFrame(frame);
            }
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
        if (d->numNodes < 1 || d->numNodes > 26) {
            throw ExprError("Must specify between 1 and 26 input clips");
        }

        // 获取所有输入clip并检查格式
        for (int i = 0; i < d->numNodes; i++) {
            d->nodes[i] = vsapi->propGetNode(in, "clips", i, &err);
            const VSVideoInfo* vi = vsapi->getVideoInfo(d->nodes[i]);
            
            if (i == 0) {
                d->vi = vsapi->getVideoInfo(d->nodes[0]);
            } else if (!checkVideoFormats(d->vi, vi)) {
                throw ExprError("All inputs must have the same format and dimensions");
            }
        }

        // 获取表达式
        int num_expr = vsapi->propNumElements(in, "expr");
        if (num_expr < 1) {
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
            fmParallelRequests, 0, d.release(), core);
    }
    catch (const std::exception& e) {
        vsapi->setError(out, ("Expr: " + std::string(e.what())).c_str());
        return;
    }
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin) {
    configFunc("com.yuygfgg.expr2", "expr",
        "VapourSynth Expression Evaluation Plugin",
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Eval",
        "clips:clip[];expr:data[];",
        exprCreate, nullptr, plugin);
}

