.class Ld0/a$a;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Ljava/lang/Runnable;


# annotations
.annotation system Ldalvik/annotation/EnclosingMethod;
    value = Ld0/a;->c(Landroid/content/SharedPreferences$Editor;Lz/i$d;)V
.end annotation

.annotation system Ldalvik/annotation/InnerClass;
    accessFlags = 0x0
    name = null
.end annotation


# instance fields
.field final synthetic a:Landroid/content/SharedPreferences$Editor;

.field final synthetic b:Lz/i$d;

.field final synthetic c:Ld0/a;


# direct methods
.method constructor <init>(Ld0/a;Landroid/content/SharedPreferences$Editor;Lz/i$d;)V
    .locals 0

    iput-object p1, p0, Ld0/a$a;->c:Ld0/a;

    iput-object p2, p0, Ld0/a$a;->a:Landroid/content/SharedPreferences$Editor;

    iput-object p3, p0, Ld0/a$a;->b:Lz/i$d;

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method


# virtual methods
.method public run()V
    .locals 3

    iget-object v0, p0, Ld0/a$a;->a:Landroid/content/SharedPreferences$Editor;

    invoke-interface {v0}, Landroid/content/SharedPreferences$Editor;->commit()Z

    move-result v0

    iget-object v1, p0, Ld0/a$a;->c:Ld0/a;

    invoke-static {v1}, Ld0/a;->b(Ld0/a;)Landroid/os/Handler;

    move-result-object v1

    new-instance v2, Ld0/a$a$a;

    invoke-direct {v2, p0, v0}, Ld0/a$a$a;-><init>(Ld0/a$a;Z)V

    invoke-virtual {v1, v2}, Landroid/os/Handler;->post(Ljava/lang/Runnable;)Z

    return-void
.end method
