.class public Lc0/a;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Lr/a;
.implements Lz/i$c;


# annotations
.annotation system Ldalvik/annotation/MemberClasses;
    value = {
        Lc0/a$e;,
        Lc0/a$b;,
        Lc0/a$d;,
        Lc0/a$c;
    }
.end annotation


# instance fields
.field private a:Landroid/content/Context;

.field private b:Lz/i;

.field private c:Lc0/a$c;


# direct methods
.method public constructor <init>()V
    .locals 0

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method

.method static synthetic c(Lc0/a;)Ljava/lang/String;
    .locals 0

    invoke-direct {p0}, Lc0/a;->j()Ljava/lang/String;

    move-result-object p0

    return-object p0
.end method

.method static synthetic d(Lc0/a;Ljava/lang/String;)Ljava/util/List;
    .locals 0

    invoke-direct {p0, p1}, Lc0/a;->m(Ljava/lang/String;)Ljava/util/List;

    move-result-object p0

    return-object p0
.end method

.method static synthetic e(Lc0/a;)Ljava/util/List;
    .locals 0

    invoke-direct {p0}, Lc0/a;->l()Ljava/util/List;

    move-result-object p0

    return-object p0
.end method

.method static synthetic g(Lc0/a;)Ljava/lang/String;
    .locals 0

    invoke-direct {p0}, Lc0/a;->n()Ljava/lang/String;

    move-result-object p0

    return-object p0
.end method

.method static synthetic h(Lc0/a;)Ljava/lang/String;
    .locals 0

    invoke-direct {p0}, Lc0/a;->k()Ljava/lang/String;

    move-result-object p0

    return-object p0
.end method

.method static synthetic i(Lc0/a;)Ljava/lang/String;
    .locals 0

    invoke-direct {p0}, Lc0/a;->o()Ljava/lang/String;

    move-result-object p0

    return-object p0
.end method

.method private j()Ljava/lang/String;
    .locals 1

    iget-object v0, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-static {v0}, Le0/a;->c(Landroid/content/Context;)Ljava/lang/String;

    move-result-object v0

    return-object v0
.end method

.method private k()Ljava/lang/String;
    .locals 1

    iget-object v0, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-static {v0}, Le0/a;->b(Landroid/content/Context;)Ljava/lang/String;

    move-result-object v0

    return-object v0
.end method

.method private l()Ljava/util/List;
    .locals 5
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "()",
            "Ljava/util/List<",
            "Ljava/lang/String;",
            ">;"
        }
    .end annotation

    new-instance v0, Ljava/util/ArrayList;

    invoke-direct {v0}, Ljava/util/ArrayList;-><init>()V

    sget v1, Landroid/os/Build$VERSION;->SDK_INT:I

    const/16 v2, 0x13

    if-lt v1, v2, :cond_1

    iget-object v1, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-virtual {v1}, Landroid/content/Context;->getExternalCacheDirs()[Ljava/io/File;

    move-result-object v1

    array-length v2, v1

    const/4 v3, 0x0

    :goto_0
    if-ge v3, v2, :cond_2

    aget-object v4, v1, v3

    if-eqz v4, :cond_0

    invoke-virtual {v4}, Ljava/io/File;->getAbsolutePath()Ljava/lang/String;

    move-result-object v4

    invoke-interface {v0, v4}, Ljava/util/List;->add(Ljava/lang/Object;)Z

    :cond_0
    add-int/lit8 v3, v3, 0x1

    goto :goto_0

    :cond_1
    iget-object v1, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-virtual {v1}, Landroid/content/Context;->getExternalCacheDir()Ljava/io/File;

    move-result-object v1

    if-eqz v1, :cond_2

    invoke-virtual {v1}, Ljava/io/File;->getAbsolutePath()Ljava/lang/String;

    move-result-object v1

    invoke-interface {v0, v1}, Ljava/util/List;->add(Ljava/lang/Object;)Z

    :cond_2
    return-object v0
.end method

.method private m(Ljava/lang/String;)Ljava/util/List;
    .locals 4
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "(",
            "Ljava/lang/String;",
            ")",
            "Ljava/util/List<",
            "Ljava/lang/String;",
            ">;"
        }
    .end annotation

    new-instance v0, Ljava/util/ArrayList;

    invoke-direct {v0}, Ljava/util/ArrayList;-><init>()V

    sget v1, Landroid/os/Build$VERSION;->SDK_INT:I

    const/16 v2, 0x13

    if-lt v1, v2, :cond_1

    iget-object v1, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-virtual {v1, p1}, Landroid/content/Context;->getExternalFilesDirs(Ljava/lang/String;)[Ljava/io/File;

    move-result-object p1

    array-length v1, p1

    const/4 v2, 0x0

    :goto_0
    if-ge v2, v1, :cond_2

    aget-object v3, p1, v2

    if-eqz v3, :cond_0

    invoke-virtual {v3}, Ljava/io/File;->getAbsolutePath()Ljava/lang/String;

    move-result-object v3

    invoke-interface {v0, v3}, Ljava/util/List;->add(Ljava/lang/Object;)Z

    :cond_0
    add-int/lit8 v2, v2, 0x1

    goto :goto_0

    :cond_1
    iget-object v1, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-virtual {v1, p1}, Landroid/content/Context;->getExternalFilesDir(Ljava/lang/String;)Ljava/io/File;

    move-result-object p1

    if-eqz p1, :cond_2

    invoke-virtual {p1}, Ljava/io/File;->getAbsolutePath()Ljava/lang/String;

    move-result-object p1

    invoke-interface {v0, p1}, Ljava/util/List;->add(Ljava/lang/Object;)Z

    :cond_2
    return-object v0
.end method

.method private n()Ljava/lang/String;
    .locals 2

    iget-object v0, p0, Lc0/a;->a:Landroid/content/Context;

    const/4 v1, 0x0

    invoke-virtual {v0, v1}, Landroid/content/Context;->getExternalFilesDir(Ljava/lang/String;)Ljava/io/File;

    move-result-object v0

    if-nez v0, :cond_0

    return-object v1

    :cond_0
    invoke-virtual {v0}, Ljava/io/File;->getAbsolutePath()Ljava/lang/String;

    move-result-object v0

    return-object v0
.end method

.method private o()Ljava/lang/String;
    .locals 1

    iget-object v0, p0, Lc0/a;->a:Landroid/content/Context;

    invoke-virtual {v0}, Landroid/content/Context;->getCacheDir()Ljava/io/File;

    move-result-object v0

    invoke-virtual {v0}, Ljava/io/File;->getPath()Ljava/lang/String;

    move-result-object v0

    return-object v0
.end method

.method private p(Lz/b;Landroid/content/Context;)V
    .locals 12

    const-string v0, "PathProviderPlugin"

    const-string v1, "plugins.flutter.io/path_provider"

    const/4 v2, 0x0

    :try_start_0
    const-class v3, Lz/i;

    const-class v4, Lz/b$c;

    invoke-virtual {p1}, Ljava/lang/Object;->getClass()Ljava/lang/Class;

    move-result-object v5

    const-string v6, "makeBackgroundTaskQueue"

    const/4 v7, 0x0

    new-array v8, v7, [Ljava/lang/Class;

    invoke-virtual {v5, v6, v8}, Ljava/lang/Class;->getMethod(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;

    move-result-object v5

    new-array v6, v7, [Ljava/lang/Object;

    invoke-virtual {v5, p1, v6}, Ljava/lang/reflect/Method;->invoke(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;

    move-result-object v5

    const/4 v6, 0x4

    new-array v8, v6, [Ljava/lang/Class;

    const-class v9, Lz/b;

    aput-object v9, v8, v7

    const-class v9, Ljava/lang/String;

    const/4 v10, 0x1

    aput-object v9, v8, v10

    const-class v9, Lz/j;

    const/4 v11, 0x2

    aput-object v9, v8, v11

    const/4 v9, 0x3

    aput-object v4, v8, v9

    invoke-virtual {v3, v8}, Ljava/lang/Class;->getConstructor([Ljava/lang/Class;)Ljava/lang/reflect/Constructor;

    move-result-object v3

    new-array v4, v6, [Ljava/lang/Object;

    aput-object p1, v4, v7

    aput-object v1, v4, v10

    sget-object v6, Lz/p;->b:Lz/p;

    aput-object v6, v4, v11

    aput-object v5, v4, v9

    invoke-virtual {v3, v4}, Ljava/lang/reflect/Constructor;->newInstance([Ljava/lang/Object;)Ljava/lang/Object;

    move-result-object v3

    check-cast v3, Lz/i;

    iput-object v3, p0, Lc0/a;->b:Lz/i;

    new-instance v3, Lc0/a$b;

    invoke-direct {v3, p0, v2}, Lc0/a$b;-><init>(Lc0/a;Lc0/a$a;)V

    iput-object v3, p0, Lc0/a;->c:Lc0/a$c;

    const-string v3, "Use TaskQueues."

    invoke-static {v0, v3}, Landroid/util/Log;->d(Ljava/lang/String;Ljava/lang/String;)I
    :try_end_0
    .catch Ljava/lang/Exception; {:try_start_0 .. :try_end_0} :catch_0

    goto :goto_0

    :catch_0
    new-instance v3, Lz/i;

    invoke-direct {v3, p1, v1}, Lz/i;-><init>(Lz/b;Ljava/lang/String;)V

    iput-object v3, p0, Lc0/a;->b:Lz/i;

    new-instance p1, Lc0/a$d;

    invoke-direct {p1, p0, v2}, Lc0/a$d;-><init>(Lc0/a;Lc0/a$a;)V

    iput-object p1, p0, Lc0/a;->c:Lc0/a$c;

    const-string p1, "Don\'t use TaskQueues."

    invoke-static {v0, p1}, Landroid/util/Log;->d(Ljava/lang/String;Ljava/lang/String;)I

    :goto_0
    iput-object p2, p0, Lc0/a;->a:Landroid/content/Context;

    iget-object p1, p0, Lc0/a;->b:Lz/i;

    invoke-virtual {p1, p0}, Lz/i;->e(Lz/i$c;)V

    return-void
.end method


# virtual methods
.method public a(Lz/h;Lz/i$d;)V
    .locals 3

    iget-object v0, p1, Lz/h;->a:Ljava/lang/String;

    invoke-virtual {v0}, Ljava/lang/String;->hashCode()I

    invoke-virtual {v0}, Ljava/lang/String;->hashCode()I

    move-result v1

    const/4 v2, -0x1

    sparse-switch v1, :sswitch_data_0

    goto :goto_0

    :sswitch_0
    const-string v1, "getTemporaryDirectory"

    invoke-virtual {v0, v1}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :cond_0

    goto :goto_0

    :cond_0
    const/4 v2, 0x5

    goto :goto_0

    :sswitch_1
    const-string v1, "getStorageDirectory"

    invoke-virtual {v0, v1}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :cond_1

    goto :goto_0

    :cond_1
    const/4 v2, 0x4

    goto :goto_0

    :sswitch_2
    const-string v1, "getApplicationDocumentsDirectory"

    invoke-virtual {v0, v1}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :cond_2

    goto :goto_0

    :cond_2
    const/4 v2, 0x3

    goto :goto_0

    :sswitch_3
    const-string v1, "getExternalStorageDirectories"

    invoke-virtual {v0, v1}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :cond_3

    goto :goto_0

    :cond_3
    const/4 v2, 0x2

    goto :goto_0

    :sswitch_4
    const-string v1, "getExternalCacheDirectories"

    invoke-virtual {v0, v1}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :cond_4

    goto :goto_0

    :cond_4
    const/4 v2, 0x1

    goto :goto_0

    :sswitch_5
    const-string v1, "getApplicationSupportDirectory"

    invoke-virtual {v0, v1}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :cond_5

    goto :goto_0

    :cond_5
    const/4 v2, 0x0

    :goto_0
    packed-switch v2, :pswitch_data_0

    invoke-interface {p2}, Lz/i$d;->c()V

    goto :goto_1

    :pswitch_0
    iget-object p1, p0, Lc0/a;->c:Lc0/a$c;

    invoke-interface {p1, p2}, Lc0/a$c;->c(Lz/i$d;)V

    goto :goto_1

    :pswitch_1
    iget-object p1, p0, Lc0/a;->c:Lc0/a$c;

    invoke-interface {p1, p2}, Lc0/a$c;->a(Lz/i$d;)V

    goto :goto_1

    :pswitch_2
    iget-object p1, p0, Lc0/a;->c:Lc0/a$c;

    invoke-interface {p1, p2}, Lc0/a$c;->e(Lz/i$d;)V

    goto :goto_1

    :pswitch_3
    const-string v0, "type"

    invoke-virtual {p1, v0}, Lz/h;->a(Ljava/lang/String;)Ljava/lang/Object;

    move-result-object p1

    check-cast p1, Ljava/lang/Integer;

    invoke-static {p1}, Lc0/i;->a(Ljava/lang/Integer;)Ljava/lang/String;

    move-result-object p1

    iget-object v0, p0, Lc0/a;->c:Lc0/a$c;

    invoke-interface {v0, p1, p2}, Lc0/a$c;->d(Ljava/lang/String;Lz/i$d;)V

    goto :goto_1

    :pswitch_4
    iget-object p1, p0, Lc0/a;->c:Lc0/a$c;

    invoke-interface {p1, p2}, Lc0/a$c;->f(Lz/i$d;)V

    goto :goto_1

    :pswitch_5
    iget-object p1, p0, Lc0/a;->c:Lc0/a$c;

    invoke-interface {p1, p2}, Lc0/a$c;->b(Lz/i$d;)V

    :goto_1
    return-void

    :sswitch_data_0
    .sparse-switch
        -0x6d37cc68 -> :sswitch_5
        -0x480b21b6 -> :sswitch_4
        0x11dc9171 -> :sswitch_3
        0x478b704f -> :sswitch_2
        0x4aadfda8 -> :sswitch_1
        0x6608ad12 -> :sswitch_0
    .end sparse-switch

    :pswitch_data_0
    .packed-switch 0x0
        :pswitch_5
        :pswitch_4
        :pswitch_3
        :pswitch_2
        :pswitch_1
        :pswitch_0
    .end packed-switch
.end method

.method public b(Lr/a$b;)V
    .locals 1

    iget-object p1, p0, Lc0/a;->b:Lz/i;

    const/4 v0, 0x0

    invoke-virtual {p1, v0}, Lz/i;->e(Lz/i$c;)V

    iput-object v0, p0, Lc0/a;->b:Lz/i;

    return-void
.end method

.method public f(Lr/a$b;)V
    .locals 1

    invoke-virtual {p1}, Lr/a$b;->b()Lz/b;

    move-result-object v0

    invoke-virtual {p1}, Lr/a$b;->a()Landroid/content/Context;

    move-result-object p1

    invoke-direct {p0, v0, p1}, Lc0/a;->p(Lz/b;Landroid/content/Context;)V

    return-void
.end method
