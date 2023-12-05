use std::fmt::Debug;

/* ----- Tokenの定義 ----- */
// ひたすら定義していく
// 最後にenum Tokenを定義してまとめる

// let
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenLet;

// mut
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMut;

// 識別子
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIdent {
    pub name: String,
}

// 数値
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenNumber {
    pub value: i64,
}

// =
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenEq;

// 演算子
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenOperator {
    pub op: String,
}

// ;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSemicolon;

// (
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenLParen;

// )
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenRParen;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Let(TokenLet),
    Mut(TokenMut),
    Ident(TokenIdent),
    Number(TokenNumber),
    Eq(TokenEq),
    Operator(TokenOperator),
    Semicolon(TokenSemicolon),
    LParen(TokenLParen),
    RParen(TokenRParen),
}

/* ----- 字句解析の実装 ----- */
// ひたすら実装していく

// 字句解析エラー
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    Msg(String),
}

// 字句解析
pub fn lex(s: &str) -> Result<Vec<Token>, LexError> {
    // 返り値のToken列
    let mut res = vec![];
    // 入力文字列をcharの列に変換しておく
    let cs: Vec<_> = s.chars().collect();

    let mut i = 0;
    while i < cs.len() {
        // 空白類はスキップ
        while i < cs.len() && cs[i].is_whitespace() {
            i += 1;
            continue;
        }

        // 最後まで来たら終了
        if i >= cs.len() {
            break;
        }

        // 記号類
        if cs[i] == '+' || cs[i] == '-' || cs[i] == '*' || cs[i] == '/' {
            res.push(Token::Operator(TokenOperator {
                op: cs[i].to_string(),
            }));
            i += 1;
            continue;
        } else if cs[i] == '(' {
            res.push(Token::LParen(TokenLParen));
            i += 1;
            continue;
        } else if cs[i] == ')' {
            res.push(Token::RParen(TokenRParen));
            i += 1;
            continue;
        } else if cs[i] == ';' {
            res.push(Token::Semicolon(TokenSemicolon));
            i += 1;
            continue;
        } else if cs[i] == '=' {
            res.push(Token::Eq(TokenEq));
            i += 1;
            continue;
        }

        // 数値
        if cs[i].is_ascii_digit() {
            let mut value = 0;
            while i < cs.len() && cs[i].is_ascii_digit() {
                value = value * 10 + cs[i].to_digit(10).unwrap() as i64;
                i += 1;
            }
            res.push(Token::Number(TokenNumber { value }));
            continue;
        }

        // 識別子 or キーワード
        if cs[i].is_ascii_alphabetic() {
            let mut ident = String::new();
            while i < cs.len() && cs[i].is_ascii_alphanumeric() {
                ident.push(cs[i]);
                i += 1;
            }
            if ident == "let" {
                res.push(Token::Let(TokenLet));
            } else if ident == "mut" {
                res.push(Token::Mut(TokenMut));
            } else {
                res.push(Token::Ident(TokenIdent { name: ident }));
            }
            continue;
        }

        // それ以外はエラーにする
        return Err(LexError::Msg(format!("invalid char: {}", cs[i])));
    }
    Ok(res)
}

/* ----- 構文解析用trait ----- */
// parse関数の意味は以下の通り
// - tokens[*i..]を対象の型として構文解析してみて、成功したらOk(Some(...))を返し、*iを進める
// - tokens[*i..]を対象の型として構文解析してみて、失敗したとき、
//     - 入力全体の構文解析の失敗を意味しないのであればOk(None)を返し、*iを進めない
//     - 入力全体の構文解析の失敗を意味するのであればErr(...)を返し、*iを進めない

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Msg(String),
}

trait Parser: Sized {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError>;
}

/* ----- Token用のParser実装 ----- */
// テンプレ的に書いてく
// *iがトークン列の範囲外で呼ばれる可能性があり、*i番目の要素があるとは限らないのでgetを使う

impl Parser for TokenLet {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Let(_)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(TokenLet));
        }
        Ok(None)
    }
}

impl Parser for TokenMut {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Mut(_)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(TokenMut));
        }
        Ok(None)
    }
}

impl Parser for TokenIdent {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Ident(ident)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(ident.clone()));
        }
        Ok(None)
    }
}

impl Parser for TokenNumber {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Number(number)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(number.clone()));
        }
        Ok(None)
    }
}

impl Parser for TokenEq {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Eq(_)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(TokenEq));
        }
        Ok(None)
    }
}

impl Parser for TokenOperator {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Operator(op)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(op.clone()));
        }
        Ok(None)
    }
}

impl Parser for TokenSemicolon {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::Semicolon(_)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(TokenSemicolon));
        }
        Ok(None)
    }
}

impl Parser for TokenLParen {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::LParen(_)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(TokenLParen));
        }
        Ok(None)
    }
}

impl Parser for TokenRParen {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(Token::RParen(_)) = tokens.get(*i) {
            *i += 1;
            return Ok(Some(TokenRParen));
        }
        Ok(None)
    }
}

/* ----- Trees that Grow 的なもの ----- */
// 元ネタ: https://www.microsoft.com/en-us/research/uploads/prod/2016/11/trees-that-grow.pdf
// 構文解析より後のフェーズにおいて、構文木に情報を付加するのに使う
// 例: 名前解決をして全体で一意のIDを降る
// Extの数だけtraitを作っても実装できるが、whereのtrait境界を連れ回すのが面倒になる(はず)ので、Extをまとめたtraitを作っている
// trait名としてDecoratorが適切かは不明

pub trait Decorator {
    type StatementLetExt: Debug + Clone + PartialEq + Eq;
    type StatementCallExt: Debug + Clone + PartialEq + Eq;
    type ExprIdentExt: Debug + Clone + PartialEq + Eq;
    type ExprNumberExt: Debug + Clone + PartialEq + Eq;
    type ExprBinOpExt: Debug + Clone + PartialEq + Eq;
    type StatementsExt: Debug + Clone + PartialEq + Eq;
}

// 修飾なし (un-decorated)
// パースした結果はこれで返す
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UD;

impl Decorator for UD {
    type StatementLetExt = ();
    type StatementCallExt = ();
    type ExprIdentExt = ();
    type ExprNumberExt = ();
    type ExprBinOpExt = ();
    type StatementsExt = ();
}

/* ----- 構文解析用のデータ構造 ----- */
// 並びはstructのfieldで並べていく (+ Decoratorのext)
// 選択はenumで選択肢を列挙していく
// ゼロ個以上・1個以上はVec<_>で表現する
// 省略可能はOption<_>で表現する

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementLet<D: Decorator> {
    pub token_let: TokenLet,
    pub token_mut: Option<TokenMut>,
    pub token_ident: TokenIdent,
    pub token_eq: TokenEq,
    pub expr: Expr<D>,
    pub token_semicolon: TokenSemicolon,
    pub ext: D::StatementLetExt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementCall<D: Decorator> {
    pub token_ident: TokenIdent,
    pub token_lparen: TokenLParen,
    pub expr: Expr<D>,
    pub token_rparen: TokenRParen,
    pub token_semicolon: TokenSemicolon,
    pub ext: D::StatementCallExt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement<D: Decorator> {
    Let(StatementLet<D>),
    Call(StatementCall<D>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprIdent<D: Decorator> {
    pub token_ident: TokenIdent,
    pub ext: D::ExprIdentExt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprNumber<D: Decorator> {
    pub token_number: TokenNumber,
    pub ext: D::ExprNumberExt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprBinOp<D: Decorator> {
    pub lhs: Box<Expr<D>>,
    pub token_op: TokenOperator,
    pub rhs: Box<Expr<D>>,
    pub ext: D::ExprBinOpExt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr<D: Decorator> {
    Ident(ExprIdent<D>),
    Number(ExprNumber<D>),
    BinOp(ExprBinOp<D>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<D: Decorator> {
    pub statements: Vec<Statement<D>>,
    pub ext: D::StatementsExt,
}

/* ----- 構文解析用のデータ構造のParserの実装 (式以外) ----- */
// 並びでOptionがついていない場合: `let Some(...) = ...::parse(tokens, &mut k)? else { ... }`
// 並びでOptionがついている場合: `let ... = ...::parse(tokens, &mut k)?;``
// 繰り返し: `let mut ... = vec![]; while let Some(...) = ...::parse(tokens, &mut k)? { ... }`
// - 1個以上の場合はwhile文の後で長さチェックを入れる

impl Parser for StatementLet<UD> {
    // 'let' 'mut'? ident '=' expr ';'
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        // 失敗したときに`*i`を進めないようにするために、一時変数kを使う
        let mut k = *i;

        let Some(token_let) = TokenLet::parse(tokens, &mut k)? else {
            return Ok(None);
        };

        // ?のときはSome(...)のlet-elseなしでOK
        let token_mut = TokenMut::parse(tokens, &mut k)?;

        let Some(token_ident) = TokenIdent::parse(tokens, &mut k)? else {
            // letで始まっていて読み進めた後なので、identがないのは入力全体の失敗を意味する (これ以降も同様)
            return Err(ParseError::Msg(format!(
                "expected ident, but found {:?}",
                tokens[k]
            )));
        };

        let Some(token_eq) = TokenEq::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected '=', but found {:?}",
                tokens[k]
            )));
        };

        let Some(expr) = Expr::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected expr, but found {:?}",
                tokens[k]
            )));
        };

        let Some(token_semicolon) = TokenSemicolon::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected ';', but found {:?}",
                tokens[k]
            )));
        };

        *i = k;
        Ok(Some(StatementLet {
            token_let,
            token_mut,
            token_ident,
            token_eq,
            expr,
            token_semicolon,
            ext: (),
        }))
    }
}

impl Parser for StatementCall<UD> {
    // ident '(' expr ')' ';'
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        let mut k = *i;

        let Some(token_ident) = TokenIdent::parse(tokens, &mut k)? else {
            return Ok(None);
        };

        let Some(token_lparen) = TokenLParen::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected '(', but found {:?}",
                tokens[k]
            )));
        };

        let Some(expr) = Expr::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected expr, but found {:?}",
                tokens[k]
            )));
        };

        let Some(token_rparen) = TokenRParen::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected ')', but found {:?}",
                tokens[k]
            )));
        };

        let Some(token_semicolon) = TokenSemicolon::parse(tokens, &mut k)? else {
            return Err(ParseError::Msg(format!(
                "expected ';', but found {:?}",
                tokens[k]
            )));
        };

        *i = k;
        Ok(Some(StatementCall {
            token_ident,
            token_lparen,
            expr,
            token_rparen,
            token_semicolon,
            ext: (),
        }))
    }
}

impl Parser for Statement<UD> {
    // statement_let / statement_call
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(statement_let) = StatementLet::parse(tokens, i)? {
            return Ok(Some(Statement::Let(statement_let)));
        }

        if let Some(statement_call) = StatementCall::parse(tokens, i)? {
            return Ok(Some(Statement::Call(statement_call)));
        }

        // statementの繰り返しの終わりの場合があるので、入力全体の失敗を意味しない
        Ok(None)
    }
}

impl Parser for ExprIdent<UD> {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        let mut k = *i;

        let Some(token_ident) = TokenIdent::parse(tokens, &mut k)? else {
            return Ok(None);
        };

        *i = k;
        Ok(Some(ExprIdent {
            token_ident,
            ext: (),
        }))
    }
}

impl Parser for ExprNumber<UD> {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        let mut k = *i;

        let Some(token_number) = TokenNumber::parse(tokens, &mut k)? else {
            return Ok(None);
        };

        *i = k;
        Ok(Some(ExprNumber {
            token_number,
            ext: (),
        }))
    }
}

impl Parser for Program<UD> {
    // statement*
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        let mut k = *i;

        let mut statements = vec![];
        while let Some(statement) = Statement::parse(tokens, &mut k)? {
            statements.push(statement);
        }

        *i = k;
        Ok(Some(Program {
            statements,
            ext: (),
        }))
    }
}

/* ----- 構文解析用のデータ構造のParserの実装 (式) ----- */
// 演算子ごとにやってもいいが、ここでは Precedence Climbing を使う
// 参考: https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing

// 演算子の結合の向き
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Assoc {
    Left,
    None,
    Right,
}

// 演算子一覧、優先順位と結合の向きを持つ
const OPS: &[(&str, i32, Assoc)] = &[
    ("+", 1, Assoc::Left),
    ("-", 1, Assoc::Left),
    ("*", 2, Assoc::Left),
    ("/", 2, Assoc::Left),
];

// 二項演算の葉ノード用の構文解析用のデータ構造、Precedence Climbing の実装をスッキリさせるのに使う
// 外には出さない(Exprに直して返す)
#[derive(Debug, Clone, PartialEq, Eq)]
enum Atom<D: Decorator> {
    Ident(ExprIdent<D>),
    Number(ExprNumber<D>),
}

impl From<Atom<UD>> for Expr<UD> {
    fn from(atom: Atom<UD>) -> Expr<UD> {
        match atom {
            Atom::Ident(expr_ident) => Expr::Ident(expr_ident),
            Atom::Number(expr_number) => Expr::Number(expr_number),
        }
    }
}

fn precedence_climbing(
    tokens: &[Token],
    i: &mut usize,
    prev: Option<(i32, Assoc)>,
) -> Result<Option<Expr<UD>>, ParseError> {
    let mut k = *i;

    let Some(atom) = Atom::parse(tokens, &mut k)? else {
        return Ok(None);
    };
    let mut expr = atom.into();

    while let Some(op) = TokenOperator::parse(tokens, &mut k)? {
        let Some((_, prec, assoc)) = OPS.iter().find(|(op2, _, _)| op2 == &op.op) else {
            return Err(ParseError::Msg(format!("unknown operator: {}", op.op)));
        };
        if let Some((prev_prec, prev_assoc)) = prev {
            // 結合がない演算子が連続するときはエラーにする
            if prev_assoc == Assoc::None && *assoc == Assoc::None {
                return Err(ParseError::Msg(format!(
                    "operator {:?} is not associative",
                    op.op
                )));
            }
            // breakするとき = 今いる場所の左側で一度式をまとめる必要がある
            // 1. 左側にいる演算子の優先順位が今見ている演算子より高いとき
            // 2. 左側にいる演算子の優先順位が今見ている演算子と同じで、結合が左結合のとき
            if prev_prec > *prec || (prev_prec == *prec && *assoc == Assoc::Left) {
                break;
            }
        }
        let Some(rhs) = precedence_climbing(tokens, &mut k, Some((*prec, *assoc)))? else {
            return Err(ParseError::Msg(format!(
                "expected rhs, but found {:?}",
                tokens[k]
            )));
        };
        expr = Expr::BinOp(ExprBinOp {
            lhs: Box::new(expr),
            token_op: op,
            rhs: Box::new(rhs),
            ext: (),
        });
    }

    *i = k;
    Ok(Some(expr))
}

impl Parser for Atom<UD> {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        if let Some(expr_ident) = ExprIdent::parse(tokens, i)? {
            return Ok(Some(Atom::Ident(expr_ident)));
        }

        if let Some(expr_number) = ExprNumber::parse(tokens, i)? {
            return Ok(Some(Atom::Number(expr_number)));
        }

        Err(ParseError::Msg(format!(
            "expected expr_ident or expr_number, but found {:?}",
            tokens[*i]
        )))
    }
}

impl Parser for Expr<UD> {
    fn parse(tokens: &[Token], i: &mut usize) -> Result<Option<Self>, ParseError> {
        precedence_climbing(tokens, i, None)
    }
}

/* ----- 入力の構文解析 ----- */

fn main() {
    let s = r#"
let mut x = 1;
let y = 2;
print(x + 3 * y);
"#;
    let tokens = lex(s).unwrap();
    println!("{:?}", tokens);

    let mut i = 0;
    let program = Program::parse(&tokens, &mut i).unwrap().unwrap();
    if i != tokens.len() {
        panic!("failed to parse");
    }
    println!("{:?}", program);
}
