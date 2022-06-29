import sys
import json
import os
from io import TextIOWrapper

from antlr4 import *
from antlr4.tree.Tree import TerminalNodeImpl

from antlr_dist.Python3Lexer import Python3Lexer
from antlr_dist.Python3Parser import Python3Parser
from antlr_dist.Python3Listener import Python3Listener


identifiersRulesNames = set(("NAME", "STRING_LITERAL", "BYTES_LITERAL", "DECIMAL_INTEGER", "OCT_INTEGER",
                            "HEX_INTEGER", "BIN_INTERGET", "FLOAT_NUMBER", "IMAG_NUMBER", "NONE", "TRUE", "FALSE"))

localVarContexts = set(("atom", "except_clause", "lambdef", "lambdef_nocond"))

MAX_DEPTH = 1000


class FuncdefProcesser(Python3Listener):
    def enterFuncdef(self, ctx: ParserRuleContext):
        func_name = ctx.NAME().getText()
        print("enter function: " + func_name)

        return super().enterFuncdef(ctx)

    def exitFuncdef(self, ctx: ParserRuleContext):
        func_name = ctx.NAME().getText()
        print("exit function: " + func_name)
        return super().exitFuncdef(ctx)


class SimpleTreeConverter:
    def __init__(self, parser: Python3Parser, tokens: CommonTokenStream, output: TextIOWrapper) -> None:
        self.parser = parser
        self.tokens = tokens
        self.output = output
        self.thisFuncName = ""
        self.stackDepth = 0
        self.beginLine = 0
        self.endLine = 0
        self.childHasLeaf = False
        self.whitespace_tmp = ""

    def dumpFuncAst(self, simpleTree: list):
        if(len(simpleTree) == 2):
            simpleTree = simpleTree[1]
        tmp = dict()
        tmp["method"] = self.thisFuncName
        tmp["beginline"] = self.beginLine
        tmp["endline"] = self.endLine
        tmp["ast"] = simpleTree
        print(json.dumps(tmp), file=self.output)

    def getLeading(self, node: TerminalNodeImpl):
        lastIndexOfToken = node.getSymbol().tokenIndex
        HIDDEN = 1
        if lastIndexOfToken < 0:
            return ''
        ws = self.tokens.getHiddenTokensToLeft(lastIndexOfToken, HIDDEN)
        if ws is None:
            return ''
        else:
            return ''.join([wst.text for wst in ws])

    def getSerializedTree(self, tree: ParserRuleContext) -> list:
        thisRuleName = ""
        hasLeaf = False
        label_temp = ""
        simpleTree: list = []

        def getRuleName(tree: ParserRuleContext) -> str:
            ruleIndex = tree.getRuleIndex()
            return self.parser.ruleNames[ruleIndex]

        def extract_terminalNode(t: TerminalNodeImpl) -> None:
            token_text = t.getText()
            nonlocal label_temp, simpleTree, hasLeaf
            if token_text != "<EOF>":
                thisToken: Token = t.getSymbol()
                tokenRuleName = self.parser.symbolicNames[thisToken.type]
                if tokenRuleName == 'NEWLINE':
                    self.whitespace_tmp += token_text
                else:
                    tok = dict()
                    if tokenRuleName == 'INDENT' or tokenRuleName == 'DEDENT':
                        tok["token"] = tokenRuleName
                    else:
                        tok["token"] = token_text
                    tok["leading"] = self.whitespace_tmp + self.getLeading(t)
                    self.whitespace_tmp = ''
                    tok["token_type"] = tokenRuleName
                    if tokenRuleName in identifiersRulesNames:
                        if thisRuleName in localVarContexts:
                            tok["var"] = True
                            if thisRuleName == "atom" and t.parentCtx.parentCtx.trailer() is not None \
                                and len(t.parentCtx.parentCtx.trailer()) > 0 \
                                and t.parentCtx.parentCtx.trailer()[0].OPEN_PAREN() is not None \
                                or token_text == "self":
                                # 例外：atom规则下的NAME紧跟函数调用; NAME为self引用
                                tok.pop("var")
                        tok["leaf"] = True
                        label_temp += "#"
                        hasLeaf = True
                    else:
                        label_temp += tok["token"]
                    tok["line"] = thisToken.line
                    self.endLine = thisToken.line
                    simpleTree.append(tok)

        def extract_internalNode(t: ParserRuleContext) -> None:
            child = self.getSerializedTree(t)
            nonlocal simpleTree, label_temp, hasLeaf
            if child is not None and len(child) > 0:
                if len(child) == 2:  # 只含一个结点的子树
                    simpleTree.append(child[1])
                    label_temp += child[0]["label"]
                    hasLeaf = hasLeaf or self.childHasLeaf
                elif not self.childHasLeaf:  # 后代都没有叶结点（叶结点为非关键字token，关键字token在内部结点中已表示）
                    label_temp += child[0]["label"]
                    for j in range(1, len(child)):
                        simpleTree.append(child[j])
                else:  # 含叶节点（非关键词token）的子树
                    label_temp += "#"
                    hasLeaf = True
                    simpleTree.append(child)

        def run() -> list:
            nonlocal hasLeaf, thisRuleName, simpleTree, label_temp
            self.stackDepth = self.stackDepth + 1
            n = tree.getChildCount()
            hasLeaf = False
            if n == 0 or self.stackDepth > MAX_DEPTH:
                self.childHasLeaf = False
                self.stackDepth = self.stackDepth - 1
                return None

            thisRuleName = getRuleName(tree)
            if thisRuleName == "funcdef":
                oldFuncName = self.thisFuncName
                self.thisFuncName = tree.getChild(1).getText()
                oldBeginLine = self.beginLine
                self.beginLine = tree.getChild(1).getSymbol().line
            start = dict()
            start["label"] = ""
            simpleTree.append(start)
            label_temp = str()
            for t in tree.getChildren():
                if isinstance(t, TerminalNodeImpl):
                    extract_terminalNode(t)
                else:
                    extract_internalNode(t)
            start["label"] = label_temp
            self.childHasLeaf = hasLeaf
            if thisRuleName == "suite" and getRuleName(tree.parentCtx) == "funcdef":
                self.dumpFuncAst(simpleTree)
            if thisRuleName == "funcdef":
                self.thisFuncName = oldFuncName
                self.beginLine = oldBeginLine
            self.stackDepth = self.stackDepth - 1
            return simpleTree

        return run()


def execute(inputFile : str, outputFile: str):
    input_stream = FileStream(inputFile, encoding='utf-8')
    lexer = Python3Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = Python3Parser(stream)
    # vocab = [token for token in lexer.getAllTokens()]
    # with open ('./tokens_debug/' + inputFile.split('/')[-1] + '.token', 'w') as f:
    #     for item in vocab:
    #         print("{0}, {1}\n".format(item, parser.symbolicNames[item.type]), file = f, flush=True)
    tree = parser.file_input()
    output_file = open(outputFile, mode='a')
    converter = SimpleTreeConverter(parser, stream, output_file)
    converter.getSerializedTree(tree)
    output_file.close()
    # processer = FuncdefProcesser()
    # walker = ParseTreeWalker()
    # walker.walk(processer, tree)


if __name__ == '__main__':
    execute(sys.argv[1], "ast_example.json")
