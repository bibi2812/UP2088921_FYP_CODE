.class public final synthetic Lc0/b;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Ljava/lang/Runnable;


# instance fields
.field public final synthetic a:Lcom/google/common/util/concurrent/f;

.field public final synthetic b:Ljava/util/concurrent/Callable;


# direct methods
.method public synthetic constructor <init>(Lcom/google/common/util/concurrent/f;Ljava/util/concurrent/Callable;)V
    .locals 0

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    iput-object p1, p0, Lc0/b;->a:Lcom/google/common/util/concurrent/f;

    iput-object p2, p0, Lc0/b;->b:Ljava/util/concurrent/Callable;

    return-void
.end method


# virtual methods
.method public final run()V
    .locals 2

    iget-object v0, p0, Lc0/b;->a:Lcom/google/common/util/concurrent/f;

    iget-object v1, p0, Lc0/b;->b:Ljava/util/concurrent/Callable;

    invoke-static {v0, v1}, Lc0/a$d;->g(Lcom/google/common/util/concurrent/f;Ljava/util/concurrent/Callable;)V

    return-void
.end method
