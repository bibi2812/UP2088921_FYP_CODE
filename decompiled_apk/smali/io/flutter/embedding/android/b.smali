.class public final Lio/flutter/embedding/android/b;
.super Ljava/lang/Object;
.source ""

# interfaces
.implements Lio/flutter/embedding/android/q;


# annotations
.annotation system Ldalvik/annotation/MemberClasses;
    value = {
        Lio/flutter/embedding/android/b$b;
    }
.end annotation

.annotation runtime Ljava/lang/Deprecated;
.end annotation


# instance fields
.field private final a:Landroid/graphics/drawable/Drawable;

.field private final b:Landroid/widget/ImageView$ScaleType;

.field private final c:J

.field private d:Lio/flutter/embedding/android/b$b;


# direct methods
.method public constructor <init>(Landroid/graphics/drawable/Drawable;)V
    .locals 3

    sget-object v0, Landroid/widget/ImageView$ScaleType;->FIT_XY:Landroid/widget/ImageView$ScaleType;

    const-wide/16 v1, 0x1f4

    invoke-direct {p0, p1, v0, v1, v2}, Lio/flutter/embedding/android/b;-><init>(Landroid/graphics/drawable/Drawable;Landroid/widget/ImageView$ScaleType;J)V

    return-void
.end method

.method public constructor <init>(Landroid/graphics/drawable/Drawable;Landroid/widget/ImageView$ScaleType;J)V
    .locals 0

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    iput-object p1, p0, Lio/flutter/embedding/android/b;->a:Landroid/graphics/drawable/Drawable;

    iput-object p2, p0, Lio/flutter/embedding/android/b;->b:Landroid/widget/ImageView$ScaleType;

    iput-wide p3, p0, Lio/flutter/embedding/android/b;->c:J

    return-void
.end method


# virtual methods
.method public a(Ljava/lang/Runnable;)V
    .locals 3

    iget-object v0, p0, Lio/flutter/embedding/android/b;->d:Lio/flutter/embedding/android/b$b;

    if-nez v0, :cond_0

    invoke-interface {p1}, Ljava/lang/Runnable;->run()V

    return-void

    :cond_0
    invoke-virtual {v0}, Landroid/widget/ImageView;->animate()Landroid/view/ViewPropertyAnimator;

    move-result-object v0

    const/4 v1, 0x0

    invoke-virtual {v0, v1}, Landroid/view/ViewPropertyAnimator;->alpha(F)Landroid/view/ViewPropertyAnimator;

    move-result-object v0

    iget-wide v1, p0, Lio/flutter/embedding/android/b;->c:J

    invoke-virtual {v0, v1, v2}, Landroid/view/ViewPropertyAnimator;->setDuration(J)Landroid/view/ViewPropertyAnimator;

    move-result-object v0

    new-instance v1, Lio/flutter/embedding/android/b$a;

    invoke-direct {v1, p0, p1}, Lio/flutter/embedding/android/b$a;-><init>(Lio/flutter/embedding/android/b;Ljava/lang/Runnable;)V

    invoke-virtual {v0, v1}, Landroid/view/ViewPropertyAnimator;->setListener(Landroid/animation/Animator$AnimatorListener;)Landroid/view/ViewPropertyAnimator;

    return-void
.end method

.method public synthetic b()Z
    .locals 1

    invoke-static {p0}, Lio/flutter/embedding/android/p;->a(Lio/flutter/embedding/android/q;)Z

    move-result v0

    return v0
.end method

.method public c(Landroid/content/Context;Landroid/os/Bundle;)Landroid/view/View;
    .locals 1

    new-instance p2, Lio/flutter/embedding/android/b$b;

    invoke-direct {p2, p1}, Lio/flutter/embedding/android/b$b;-><init>(Landroid/content/Context;)V

    iput-object p2, p0, Lio/flutter/embedding/android/b;->d:Lio/flutter/embedding/android/b$b;

    iget-object p1, p0, Lio/flutter/embedding/android/b;->a:Landroid/graphics/drawable/Drawable;

    iget-object v0, p0, Lio/flutter/embedding/android/b;->b:Landroid/widget/ImageView$ScaleType;

    invoke-virtual {p2, p1, v0}, Lio/flutter/embedding/android/b$b;->a(Landroid/graphics/drawable/Drawable;Landroid/widget/ImageView$ScaleType;)V

    iget-object p1, p0, Lio/flutter/embedding/android/b;->d:Lio/flutter/embedding/android/b$b;

    return-object p1
.end method

.method public synthetic d()Landroid/os/Bundle;
    .locals 1

    invoke-static {p0}, Lio/flutter/embedding/android/p;->b(Lio/flutter/embedding/android/q;)Landroid/os/Bundle;

    move-result-object v0

    return-object v0
.end method
