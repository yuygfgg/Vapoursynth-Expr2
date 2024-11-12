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

class ExprError : public std::exception {
private:
    std::string msg;
public:
    ExprError(const std::string& message) : msg(message) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

// 栈操作封装
class ExprStack {
private:
    std::stack<float> stack;
    std::vector<float> temp_storage;  // 用于临时存储操作数

    void getOne(float& value) {
        if (stack.empty()) {
            throw ExprError("Stack underflow in basic operation");
        }
        value = stack.top();
        stack.pop();
    }

public:
    template<typename... Args>
    void getOperands(Args&... args) {
        if (stack.size() < sizeof...(Args)) {
            throw std::runtime_error("Stack underflow");
        }
        (getOne(args), ...);
    }

    void pushResult(float result) {
        stack.push(result);
    }

    float peekAt(size_t n) const {
        if (stack.size() <= n) {
            throw std::runtime_error("Stack underflow in peek operation");
        }
        
        auto temp = stack;
        for (size_t i = 0; i < n; ++i) {
            temp.pop();
        }
        return temp.top();
    }

    void swapN(size_t n) {
        if (n == 0) {
            throw ExprError("swap index must be greater than 0");
        }
        if (stack.size() <= n) {
            throw ExprError("Stack underflow in swap" + std::to_string(n) + 
                            ": need " + std::to_string(n + 1) + " values");
        }

        // 清空临时存储
        temp_storage.clear();
        
        // 保存top到n位置的值
        for (size_t i = 0; i <= n; ++i) {
            temp_storage.push_back(stack.top());
            stack.pop();
        }
        
        // 按照正确的顺序放回
        // 首先放回原来的top值
        stack.push(temp_storage[0]);
        
        // 然后是中间的值（如果有的话）
        for (size_t i = temp_storage.size() - 1; i > 1; --i) {
            stack.push(temp_storage[i]);
        }
        
        // 最后放回要交换的值
        stack.push(temp_storage[1]);
    }

    void dupN(size_t n) {
        if (stack.size() <= n) {
            throw ExprError("Stack underflow in dup" + std::to_string(n) + 
                            ": need " + std::to_string(n + 1) + " values");
        }
        
        // 保存需要的值
        temp_storage.clear();
        for (size_t i = 0; i < n + 1; ++i) {
            temp_storage.push_back(stack.top());
            stack.pop();
        }
        
        // 先将值放回去
        for (auto it = temp_storage.rbegin(); it != temp_storage.rend(); ++it) {
            stack.push(*it);
        }
        
        // 然后将要复制的值压入栈顶
        stack.push(temp_storage[temp_storage.size() - 1]);
    }

    template<typename Func>
    void unaryOp(Func op) {
        float a;
        getOperands(a);
        pushResult(op(a));
    }

    template<typename Func>
    void binaryOp(Func op) {
        float b, a;
        getOperands(b, a);
        pushResult(op(a, b));
    }

    template<typename Func>
    void ternaryOp(Func op) {
        float c, b, a;
        getOperands(c, b, a);
        pushResult(op(a, b, c));
    }

    float getResult() const {
        if (stack.size() != 1) {
            throw std::runtime_error("Invalid expression evaluation");
        }
        return stack.top();
    }

    void push(float value) {
        stack.push(value);
    }

    void clear() {
        while (!stack.empty()) {
            stack.pop();
        }
    }

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

// 操作符管理
class OperatorManager {
private:
    using OperatorFunc = std::function<void(ExprStack&)>;
    std::map<std::string, OperatorFunc> operators;

public:
    void registerUnaryOp(const std::string& token, std::function<float(float)> func) {
        operators[token] = [func](ExprStack& stack) {
            stack.unaryOp(func);
        };
    }

    void registerBinaryOp(const std::string& token, std::function<float(float, float)> func) {
        operators[token] = [func](ExprStack& stack) {
            stack.binaryOp(func);
        };
    }

    void registerTernaryOp(const std::string& token, std::function<float(float, float, float)> func) {
        operators[token] = [func](ExprStack& stack) {
            stack.ternaryOp(func);
        };
    }

    bool hasOperator(const std::string& token) const {
        return operators.find(token) != operators.end();
    }

    void executeOperator(const std::string& token, ExprStack& stack) const {
        auto it = operators.find(token);
        if (it != operators.end()) {
            it->second(stack);
        } else {
            throw std::runtime_error("Unknown operator: " + token);
        }
    }

    void initDefaultOperators() {
        // 基本算术运算符
        registerBinaryOp("+", [](float a, float b) { return a + b; });
        registerBinaryOp("-", [](float a, float b) { return a - b; });
        registerBinaryOp("*", [](float a, float b) { return a * b; });
        registerBinaryOp("/", [](float a, float b) { 
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b; 
        });

        // 数学函数
        registerUnaryOp("exp", [](float a) { 
            if (a > 88) throw std::runtime_error("exp overflow");
            return std::exp(a); 
        });
        registerUnaryOp("log", [](float a) { 
            if (a <= 0) throw std::runtime_error("log domain error");
            return std::log(a); 
        });
        registerUnaryOp("sqrt", [](float a) { 
            if (a < 0) throw std::runtime_error("sqrt domain error");
            return std::sqrt(a); 
        });
        registerUnaryOp("sin", [](float a) { return std::sin(a); });
        registerUnaryOp("cos", [](float a) { return std::cos(a); });
        registerUnaryOp("abs", [](float a) { return std::abs(a); });
        registerUnaryOp("not", [](float a) { return a > 0 ? 0.0f : 1.0f; });

        // 比较运算符
        registerBinaryOp(">", [](float a, float b) { return a > b ? 1.0f : 0.0f; });
        registerBinaryOp("<", [](float a, float b) { return a < b ? 1.0f : 0.0f; });
        registerBinaryOp("=", [](float a, float b) { return a == b ? 1.0f : 0.0f; });
        registerBinaryOp(">=", [](float a, float b) { return a >= b ? 1.0f : 0.0f; });
        registerBinaryOp("<=", [](float a, float b) { return a <= b ? 1.0f : 0.0f; });

        // 逻辑运算符
        registerBinaryOp("and", [](float a, float b) { return (a > 0 && b > 0) ? 1.0f : 0.0f; });
        registerBinaryOp("or", [](float a, float b) { return (a > 0 || b > 0) ? 1.0f : 0.0f; });
        registerBinaryOp("xor", [](float a, float b) { 
            return ((a > 0) != (b > 0)) ? 1.0f : 0.0f; 
        });

        // 其他数学运算符
        registerBinaryOp("max", [](float a, float b) { return std::max(a, b); });
        registerBinaryOp("min", [](float a, float b) { return std::min(a, b); });
        registerBinaryOp("pow", [](float a, float b) { 
            if (a < 0) throw std::runtime_error("pow domain error");
            float result = std::pow(a, b);
            if (result > 3e38f || result < 1e-38f) 
                throw std::runtime_error("pow range error");
            return result;
        });

        // 栈操作
        operators["dup"] = [](ExprStack& stack) { stack.dupN(0); };
        for (int i = 0; i <= 25; ++i) {
            operators["dup" + std::to_string(i)] = [i](ExprStack& stack) {
                stack.dupN(i);
            };
        }

        operators["swap"] = [](ExprStack& stack) { stack.swapN(1); };
        for (int i = 1; i <= 25; ++i) {
            operators["swap" + std::to_string(i)] = [i](ExprStack& stack) {
                stack.swapN(i);
            };
        }

        // 三元运算符
        registerTernaryOp("?", [](float cond, float t, float f) { 
            return cond > 0 ? t : f; 
        });
    }
};

// 语法检查器类
class ExpressionValidator {
private:
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

    static bool isOperator(const std::string& token) {
        static const std::set<std::string> operators = {
            "+", "-", "*", "/", "exp", "log", "sqrt", "sin", "cos", "abs", "not",
            ">", "<", "=", ">=", "<=", "and", "or", "xor", "max", "min", "pow", "?",
            "dup", "swap"
        };
        
        // 检查基本运算符
        if (operators.find(token) != operators.end()) return true;
        
        // 检查 dupN 和 swapN
        if (token.substr(0, 3) == "dup" || token.substr(0, 4) == "swap") {
            try {
                size_t pos;
                int n = std::stoi(token.substr(token.find_first_of("0123456789")), &pos);
                return n >= 0 && n <= 25;
            } catch (...) {
                return false;
            }
        }
        
        return false;
    }

    static void validateStackEffect(const std::vector<std::string>& tokens) {
        int stackSize = 0;
        int maxStackSize = 0;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            
            // 数字和变量入栈
            if (isNumber(token) || isVariable(token)) {
                stackSize++;
            }
            // 运算符处理
            else if (isOperator(token)) {
                if (token == "?") {
                    if (stackSize < 3) throw ExprError(
                        "Stack underflow at position " + std::to_string(i) + 
                        ": ternary operator '?' requires 3 operands");
                    stackSize -= 2;
                }
                else if (token.substr(0, 3) == "dup") {
                    if (stackSize < 1) throw ExprError(
                        "Stack underflow at position " + std::to_string(i) + 
                        ": dup requires at least 1 value on stack");
                    stackSize++;
                }
                else if (token.substr(0, 4) == "swap") {
                    if (stackSize < 2) throw ExprError(
                        "Stack underflow at position " + std::to_string(i) + 
                        ": swap requires at least 2 values on stack");
                }
                else if (token == "not" || token == "abs" || token == "exp" || 
                        token == "log" || token == "sqrt" || token == "sin" || 
                        token == "cos") {
                    if (stackSize < 1) throw ExprError(
                        "Stack underflow at position " + std::to_string(i) + 
                        ": unary operator '" + token + "' requires 1 operand");
                }
                else {
                    if (stackSize < 2) throw ExprError(
                        "Stack underflow at position " + std::to_string(i) + 
                        ": binary operator '" + token + "' requires 2 operands");
                    stackSize--;
                }
            }
            else {
                throw ExprError("Invalid token at position " + std::to_string(i) + 
                                ": '" + token + "'");
            }
            
            maxStackSize = std::max(maxStackSize, stackSize);
            if (maxStackSize > 100) throw ExprError("Stack size limit exceeded");
        }
        
        if (stackSize != 1) {
            throw ExprError("Invalid expression: should leave exactly one value on stack, " + 
                            std::to_string(stackSize) + " values found");
        }
    }

public:
    static void validate(const std::string& expr, int numClips) {
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
    OperatorManager opManager;
    ExprStack stack;

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
    ExprCalculator() {
        opManager.initDefaultOperators();
    }

    float evaluate(const std::string& expr, const std::vector<float>& values) {
        if (expr.empty()) {
            if (values.empty()) return 0.0f;
            return values[0];
        }

        try {
            stack.clear();
            std::vector<std::string> tokens = tokenize(expr);
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto& token = tokens[i];
                try {
                    if (token.substr(0, 4) == "swap") {
                        int n;
                        if (token == "swap") {
                            n = 1;
                        } else {
                            n = std::stoi(token.substr(4));
                        }
                        stack.swapN(n);
                    }
                    else if (token.substr(0, 3) == "dup") {
                        int n;
                        if (token == "dup") {
                            n = 0;
                        } else {
                            n = std::stoi(token.substr(3));
                        }
                        stack.dupN(n);
                    }
                    else if (opManager.hasOperator(token)) {
                        opManager.executeOperator(token, stack);
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
};

// VapourSynth插件结构
struct ExprData {
    VSNodeRef* nodes[26];  // 支持x,y,z + a-w共26个输入clip
    int numNodes;
    const VSVideoInfo* vi;
    std::vector<std::string> expressions;  // 每个平面一个表达式
    ExprCalculator calculator;
};

// 检查所有输入clip的格式是否兼容
static bool checkVideoFormats(const VSVideoInfo* vi1, const VSVideoInfo* vi2) {
    return vi1->format->colorFamily == vi2->format->colorFamily &&
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
                            // 对于浮点格式，UV平面需要特殊处理
                            if ((plane == 1 || plane == 2) &&
                                (fi->colorFamily == cmYUV || fi->colorFamily == cmYCoCg)) {
                                value += 0.5f;  // 从[-0.5,0.5]转换到[0,1]
                            }
                        }
                        else {
                            if (bits > 8) {
                                value = static_cast<float>(reinterpret_cast<const uint16_t*>(
                                    srcp + y * src_stride)[x]);
                            }
                            else {
                                value = static_cast<float>(srcp[y * src_stride + x]);
                            }
                        }
                        pixel_values[i] = value;
                    }

                    // 计算表达式
                    float result = d->calculator.evaluate(d->expressions[plane], pixel_values);

                    // 写入结果
                    if (isFloat) {
                        // 对于浮点格式，UV平面需要特殊处理
                        if ((plane == 1 || plane == 2) &&
                            (fi->colorFamily == cmYUV || fi->colorFamily == cmYCoCg)) {
                            result -= 0.5f;  // 转回[-0.5,0.5]范围
                        }
                        reinterpret_cast<float*>(dstp + y * dst_stride)[x] = result;
                    }
                    else {
                        result = std::clamp(result, 0.0f, static_cast<float>(maxValue));
                        if (bits > 8) {
                            reinterpret_cast<uint16_t*>(dstp + y * dst_stride)[x] =
                                static_cast<uint16_t>(result);
                        }
                        else {
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
        // 发生异常时，释放已分配的帧
        if (dst) {
            vsapi->freeFrame(dst);
        }
        for (const auto& frame : src_frames) {
            if (frame != nullptr) {
                vsapi->freeFrame(frame);
            }
        }

        // 设置错误信息并返回 nullptr
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
    std::unique_ptr<ExprData> d(new ExprData());
    int err;

    // 获取输入clips
    d->numNodes = vsapi->propNumElements(in, "clips");
    if (d->numNodes < 1 || d->numNodes > 26) {
        vsapi->setError(out, "Expr: Must specify between 1 and 26 input clips");
        return;
    }

    // 获取所有输入clip并检查格式
    for (int i = 0; i < d->numNodes; i++) {
        d->nodes[i] = vsapi->propGetNode(in, "clips", i, &err);
        const VSVideoInfo* vi = vsapi->getVideoInfo(d->nodes[i]);
        
        if (i == 0) {
            d->vi = vsapi->getVideoInfo(d->nodes[0]);
        } else if (!checkVideoFormats(d->vi, vi)) {
            for (int j = 0; j <= i; j++) {
                vsapi->freeNode(d->nodes[j]);
            }
            vsapi->setError(out, "Expr: All inputs must have the same format and dimensions");
            return;
        }
    }

    // 获取表达式
    int num_expr = vsapi->propNumElements(in, "expr");
    if (num_expr < 1) {
        for (int i = 0; i < d->numNodes; i++) {
            vsapi->freeNode(d->nodes[i]);
        }
        vsapi->setError(out, "Expr: At least one expression must be specified");
        return;
    }

    // 为每个平面获取表达式
    for (int i = 0; i < num_expr && i < d->vi->format->numPlanes; i++) {
        d->expressions.push_back(vsapi->propGetData(in, "expr", i, &err));
    }

    // 如果表达式数量少于平面数量，用最后一个表达式补充
    while (d->expressions.size() < d->vi->format->numPlanes) {
        d->expressions.push_back(d->expressions.back());
    }

    // 创建过滤器
    #if VAPOURSYNTH_API_VERSION >= 4
        vsapi->createFilter(in, out, "Expr", exprInit, exprGetFrame, exprFree, 
            fmParallel, 0, d.release(), core);
    #else
        VSFilterDependency deps[] = {{d->nodes[0], rpStrictSpatial}};
        vsapi->createFilter(in, out, "Expr", exprInit, exprGetFrame, exprFree, 
            fmParallel, deps, 1, d.release(), core);
    #endif
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin) {
    configFunc("com.yuygfgg.expr2", "expr",
        "VapourSynth Expression Evaluation Plugin",
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Eval",
        "clips:clip[];expr:data[];",
        exprCreate, nullptr, plugin);
}