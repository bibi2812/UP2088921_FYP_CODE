.class public Ly/k;
.super Ljava/lang/Object;
.source ""


# instance fields
.field public final a:Z

.field private b:[B

.field private c:Lz/i;

.field private d:Lz/i$d;

.field private e:Z

.field private f:Z

.field private final g:Lz/i$c;


# direct methods
.method public constructor <init>(Ln/a;Z)V
    .locals 3

    new-instance v0, Lz/i;

    sget-object v1, Lz/p;->b:Lz/p;

    const-string v2, "flutter/restoration"

    invoke-direct {v0, p1, v2, v1}, Lz/i;-><init>(Lz/b;Ljava/lang/String;Lz/j;)V

    invoke-direct {p0, v0, p2}, Ly/k;-><init>(Lz/i;Z)V

    return-void
.end method

.method constructor <init>(Lz/i;Z)V
    .locals 1

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    const/4 v0, 0x0

    iput-boolean v0, p0, Ly/k;->e:Z

    iput-boolean v0, p0, Ly/k;->f:Z

    new-instance v0, Ly/k$b;

    invoke-direct {v0, p0}, Ly/k$b;-><init>(Ly/k;)V

    iput-object v0, p0, Ly/k;->g:Lz/i$c;

    iput-object p1, p0, Ly/k;->c:Lz/i;

    iput-boolean p2, p0, Ly/k;->a:Z

    invoke-virtual {p1, v0}, Lz/i;->e(Lz/i$c;)V

    return-void
.end method

.method static synthetic a(Ly/k;)[B
    .locals 0

    iget-object p0, p0, Ly/k;->b:[B

    return-object p0
.end method

.method static synthetic b(Ly/k;[B)[B
    .locals 0

    iput-object p1, p0, Ly/k;->b:[B

    return-object p1
.end method

.method static synthetic c(Ly/k;Z)Z
    .locals 0

    iput-boolean p1, p0, Ly/k;->f:Z

    return p1
.end method

.method static synthetic d(Ly/k;)Z
    .locals 0

    iget-boolean p0, p0, Ly/k;->e:Z

    return p0
.end method

.method static synthetic e(Ly/k;[B)Ljava/util/Map;
    .locals 0

    invoke-direct {p0, p1}, Ly/k;->i([B)Ljava/util/Map;

    move-result-object p0

    return-object p0
.end method

.method static synthetic f(Ly/k;Lz/i$d;)Lz/i$d;
    .locals 0

    iput-object p1, p0, Ly/k;->d:Lz/i$d;

    return-object p1
.end method

.method private i([B)Ljava/util/Map;
    .locals 3
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "([B)",
            "Ljava/util/Map<",
            "Ljava/lang/String;",
            "Ljava/lang/Object;",
            ">;"
        }
    .end annotation

    new-instance v0, Ljava/util/HashMap;

    invoke-direct {v0}, Ljava/util/HashMap;-><init>()V

    sget-object v1, Ljava/lang/Boolean;->TRUE:Ljava/lang/Boolean;

    const-string v2, "enabled"

    invoke-interface {v0, v2, v1}, Ljava/util/Map;->put(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;

    const-string v1, "data"

    invoke-interface {v0, v1, p1}, Ljava/util/Map;->put(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;

    return-object v0
.end method


# virtual methods
.method public g()V
    .locals 1

    const/4 v0, 0x0

    iput-object v0, p0, Ly/k;->b:[B

    return-void
.end method

.method public h()[B
    .locals 1

    iget-object v0, p0, Ly/k;->b:[B

    return-object v0
.end method

.method public j([B)V
    .locals 3

    const/4 v0, 0x1

    iput-boolean v0, p0, Ly/k;->e:Z

    iget-object v0, p0, Ly/k;->d:Lz/i$d;

    if-eqz v0, :cond_1

    invoke-direct {p0, p1}, Ly/k;->i([B)Ljava/util/Map;

    move-result-object v1

    invoke-interface {v0, v1}, Lz/i$d;->b(Ljava/lang/Object;)V

    const/4 v0, 0x0

    iput-object v0, p0, Ly/k;->d:Lz/i$d;

    :cond_0
    iput-object p1, p0, Ly/k;->b:[B

    goto :goto_0

    :cond_1
    iget-boolean v0, p0, Ly/k;->f:Z

    if-eqz v0, :cond_0

    iget-object v0, p0, Ly/k;->c:Lz/i;

    invoke-direct {p0, p1}, Ly/k;->i([B)Ljava/util/Map;

    move-result-object v1

    new-instance v2, Ly/k$a;

    invoke-direct {v2, p0, p1}, Ly/k$a;-><init>(Ly/k;[B)V

    const-string p1, "push"

    invoke-virtual {v0, p1, v1, v2}, Lz/i;->d(Ljava/lang/String;Ljava/lang/Object;Lz/i$d;)V

    :goto_0
    return-void
.end method
