.class Li/c$i$a;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Ljava/lang/Runnable;


# annotations
.annotation system Ldalvik/annotation/EnclosingMethod;
    value = Li/c$i;->b(Ljava/lang/Object;)V
.end annotation

.annotation system Ldalvik/annotation/InnerClass;
    accessFlags = 0x0
    name = null
.end annotation


# instance fields
.field final synthetic a:Ljava/lang/Object;

.field final synthetic b:Li/c$i;


# direct methods
.method constructor <init>(Li/c$i;Ljava/lang/Object;)V
    .locals 0

    iput-object p1, p0, Li/c$i$a;->b:Li/c$i;

    iput-object p2, p0, Li/c$i$a;->a:Ljava/lang/Object;

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method


# virtual methods
.method public run()V
    .locals 2

    iget-object v0, p0, Li/c$i$a;->b:Li/c$i;

    invoke-static {v0}, Li/c$i;->d(Li/c$i;)Lz/i$d;

    move-result-object v0

    iget-object v1, p0, Li/c$i$a;->a:Ljava/lang/Object;

    invoke-interface {v0, v1}, Lz/i$d;->b(Ljava/lang/Object;)V

    return-void
.end method
