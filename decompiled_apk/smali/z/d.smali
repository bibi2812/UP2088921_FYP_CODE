.class public final Lz/d;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Lz/g;


# annotations
.annotation system Ldalvik/annotation/Signature;
    value = {
        "Ljava/lang/Object;",
        "Lz/g<",
        "Ljava/lang/Object;",
        ">;"
    }
.end annotation


# static fields
.field public static final a:Lz/d;


# direct methods
.method static constructor <clinit>()V
    .locals 1

    new-instance v0, Lz/d;

    invoke-direct {v0}, Lz/d;-><init>()V

    sput-object v0, Lz/d;->a:Lz/d;

    return-void
.end method

.method private constructor <init>()V
    .locals 0

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method


# virtual methods
.method public a(Ljava/nio/ByteBuffer;)Ljava/lang/Object;
    .locals 2

    const-string v0, "Invalid JSON"

    if-nez p1, :cond_0

    const/4 p1, 0x0

    return-object p1

    :cond_0
    :try_start_0
    sget-object v1, Lz/q;->b:Lz/q;

    invoke-virtual {v1, p1}, Lz/q;->c(Ljava/nio/ByteBuffer;)Ljava/lang/String;

    move-result-object p1

    new-instance v1, Lorg/json/JSONTokener;

    invoke-direct {v1, p1}, Lorg/json/JSONTokener;-><init>(Ljava/lang/String;)V

    invoke-virtual {v1}, Lorg/json/JSONTokener;->nextValue()Ljava/lang/Object;

    move-result-object p1

    invoke-virtual {v1}, Lorg/json/JSONTokener;->more()Z

    move-result v1

    if-nez v1, :cond_1

    return-object p1

    :cond_1
    new-instance p1, Ljava/lang/IllegalArgumentException;

    invoke-direct {p1, v0}, Ljava/lang/IllegalArgumentException;-><init>(Ljava/lang/String;)V

    throw p1
    :try_end_0
    .catch Lorg/json/JSONException; {:try_start_0 .. :try_end_0} :catch_0

    :catch_0
    move-exception p1

    new-instance v1, Ljava/lang/IllegalArgumentException;

    invoke-direct {v1, v0, p1}, Ljava/lang/IllegalArgumentException;-><init>(Ljava/lang/String;Ljava/lang/Throwable;)V

    throw v1
.end method

.method public b(Ljava/lang/Object;)Ljava/nio/ByteBuffer;
    .locals 1

    if-nez p1, :cond_0

    const/4 p1, 0x0

    return-object p1

    :cond_0
    invoke-static {p1}, Lz/f;->a(Ljava/lang/Object;)Ljava/lang/Object;

    move-result-object p1

    instance-of v0, p1, Ljava/lang/String;

    if-eqz v0, :cond_1

    sget-object v0, Lz/q;->b:Lz/q;

    check-cast p1, Ljava/lang/String;

    invoke-static {p1}, Lorg/json/JSONObject;->quote(Ljava/lang/String;)Ljava/lang/String;

    move-result-object p1

    :goto_0
    invoke-virtual {v0, p1}, Lz/q;->d(Ljava/lang/String;)Ljava/nio/ByteBuffer;

    move-result-object p1

    return-object p1

    :cond_1
    sget-object v0, Lz/q;->b:Lz/q;

    invoke-virtual {p1}, Ljava/lang/Object;->toString()Ljava/lang/String;

    move-result-object p1

    goto :goto_0
.end method
