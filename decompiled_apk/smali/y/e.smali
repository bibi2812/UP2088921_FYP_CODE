.class public Ly/e;
.super Ljava/lang/Object;
.source ""


# instance fields
.field public final a:Lz/a;
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "Lz/a<",
            "Ljava/lang/String;",
            ">;"
        }
    .end annotation
.end field


# direct methods
.method public constructor <init>(Ln/a;)V
    .locals 3

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    new-instance v0, Lz/a;

    sget-object v1, Lz/q;->b:Lz/q;

    const-string v2, "flutter/lifecycle"

    invoke-direct {v0, p1, v2, v1}, Lz/a;-><init>(Lz/b;Ljava/lang/String;Lz/g;)V

    iput-object v0, p0, Ly/e;->a:Lz/a;

    return-void
.end method


# virtual methods
.method public a()V
    .locals 2

    const-string v0, "LifecycleChannel"

    const-string v1, "Sending AppLifecycleState.detached message."

    invoke-static {v0, v1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v0, p0, Ly/e;->a:Lz/a;

    const-string v1, "AppLifecycleState.detached"

    invoke-virtual {v0, v1}, Lz/a;->c(Ljava/lang/Object;)V

    return-void
.end method

.method public b()V
    .locals 2

    const-string v0, "LifecycleChannel"

    const-string v1, "Sending AppLifecycleState.inactive message."

    invoke-static {v0, v1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v0, p0, Ly/e;->a:Lz/a;

    const-string v1, "AppLifecycleState.inactive"

    invoke-virtual {v0, v1}, Lz/a;->c(Ljava/lang/Object;)V

    return-void
.end method

.method public c()V
    .locals 2

    const-string v0, "LifecycleChannel"

    const-string v1, "Sending AppLifecycleState.paused message."

    invoke-static {v0, v1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v0, p0, Ly/e;->a:Lz/a;

    const-string v1, "AppLifecycleState.paused"

    invoke-virtual {v0, v1}, Lz/a;->c(Ljava/lang/Object;)V

    return-void
.end method

.method public d()V
    .locals 2

    const-string v0, "LifecycleChannel"

    const-string v1, "Sending AppLifecycleState.resumed message."

    invoke-static {v0, v1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v0, p0, Ly/e;->a:Lz/a;

    const-string v1, "AppLifecycleState.resumed"

    invoke-virtual {v0, v1}, Lz/a;->c(Ljava/lang/Object;)V

    return-void
.end method
