.class public final synthetic Lio/flutter/plugin/platform/j;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Landroid/view/View$OnFocusChangeListener;


# instance fields
.field public final synthetic a:Lio/flutter/plugin/platform/k$a;

.field public final synthetic b:Ly/j$b;


# direct methods
.method public synthetic constructor <init>(Lio/flutter/plugin/platform/k$a;Ly/j$b;)V
    .locals 0

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    iput-object p1, p0, Lio/flutter/plugin/platform/j;->a:Lio/flutter/plugin/platform/k$a;

    iput-object p2, p0, Lio/flutter/plugin/platform/j;->b:Ly/j$b;

    return-void
.end method


# virtual methods
.method public final onFocusChange(Landroid/view/View;Z)V
    .locals 2

    iget-object v0, p0, Lio/flutter/plugin/platform/j;->a:Lio/flutter/plugin/platform/k$a;

    iget-object v1, p0, Lio/flutter/plugin/platform/j;->b:Ly/j$b;

    invoke-static {v0, v1, p1, p2}, Lio/flutter/plugin/platform/k$a;->j(Lio/flutter/plugin/platform/k$a;Ly/j$b;Landroid/view/View;Z)V

    return-void
.end method
