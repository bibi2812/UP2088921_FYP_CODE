.class public Lio/flutter/embedding/android/i;
.super Landroid/view/TextureView;
.source ""

# interfaces
.implements Lx/c;


# instance fields
.field private a:Z

.field private b:Z

.field private c:Lx/a;

.field private d:Landroid/view/Surface;

.field private final e:Landroid/view/TextureView$SurfaceTextureListener;


# direct methods
.method public constructor <init>(Landroid/content/Context;)V
    .locals 1

    const/4 v0, 0x0

    invoke-direct {p0, p1, v0}, Lio/flutter/embedding/android/i;-><init>(Landroid/content/Context;Landroid/util/AttributeSet;)V

    return-void
.end method

.method public constructor <init>(Landroid/content/Context;Landroid/util/AttributeSet;)V
    .locals 0

    invoke-direct {p0, p1, p2}, Landroid/view/TextureView;-><init>(Landroid/content/Context;Landroid/util/AttributeSet;)V

    const/4 p1, 0x0

    iput-boolean p1, p0, Lio/flutter/embedding/android/i;->a:Z

    iput-boolean p1, p0, Lio/flutter/embedding/android/i;->b:Z

    new-instance p1, Lio/flutter/embedding/android/i$a;

    invoke-direct {p1, p0}, Lio/flutter/embedding/android/i$a;-><init>(Lio/flutter/embedding/android/i;)V

    iput-object p1, p0, Lio/flutter/embedding/android/i;->e:Landroid/view/TextureView$SurfaceTextureListener;

    invoke-direct {p0}, Lio/flutter/embedding/android/i;->n()V

    return-void
.end method

.method static synthetic d(Lio/flutter/embedding/android/i;Z)Z
    .locals 0

    iput-boolean p1, p0, Lio/flutter/embedding/android/i;->a:Z

    return p1
.end method

.method static synthetic e(Lio/flutter/embedding/android/i;)Z
    .locals 0

    iget-boolean p0, p0, Lio/flutter/embedding/android/i;->b:Z

    return p0
.end method

.method static synthetic f(Lio/flutter/embedding/android/i;)V
    .locals 0

    invoke-direct {p0}, Lio/flutter/embedding/android/i;->l()V

    return-void
.end method

.method static synthetic g(Lio/flutter/embedding/android/i;II)V
    .locals 0

    invoke-direct {p0, p1, p2}, Lio/flutter/embedding/android/i;->k(II)V

    return-void
.end method

.method static synthetic h(Lio/flutter/embedding/android/i;)V
    .locals 0

    invoke-direct {p0}, Lio/flutter/embedding/android/i;->m()V

    return-void
.end method

.method static synthetic i(Lio/flutter/embedding/android/i;)Landroid/view/Surface;
    .locals 0

    iget-object p0, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    return-object p0
.end method

.method static synthetic j(Lio/flutter/embedding/android/i;Landroid/view/Surface;)Landroid/view/Surface;
    .locals 0

    iput-object p1, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    return-object p1
.end method

.method private k(II)V
    .locals 2

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    if-eqz v0, :cond_0

    new-instance v0, Ljava/lang/StringBuilder;

    invoke-direct {v0}, Ljava/lang/StringBuilder;-><init>()V

    const-string v1, "Notifying FlutterRenderer that Android surface size has changed to "

    invoke-virtual {v0, v1}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    invoke-virtual {v0, p1}, Ljava/lang/StringBuilder;->append(I)Ljava/lang/StringBuilder;

    const-string v1, " x "

    invoke-virtual {v0, v1}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    invoke-virtual {v0, p2}, Ljava/lang/StringBuilder;->append(I)Ljava/lang/StringBuilder;

    invoke-virtual {v0}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object v0

    const-string v1, "FlutterTextureView"

    invoke-static {v1, v0}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    invoke-virtual {v0, p1, p2}, Lx/a;->s(II)V

    return-void

    :cond_0
    new-instance p1, Ljava/lang/IllegalStateException;

    const-string p2, "changeSurfaceSize() should only be called when flutterRenderer is non-null."

    invoke-direct {p1, p2}, Ljava/lang/IllegalStateException;-><init>(Ljava/lang/String;)V

    throw p1
.end method

.method private l()V
    .locals 2

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    if-eqz v0, :cond_1

    invoke-virtual {p0}, Landroid/view/TextureView;->getSurfaceTexture()Landroid/graphics/SurfaceTexture;

    move-result-object v0

    if-eqz v0, :cond_1

    iget-object v0, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    if-eqz v0, :cond_0

    invoke-virtual {v0}, Landroid/view/Surface;->release()V

    const/4 v0, 0x0

    iput-object v0, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    :cond_0
    new-instance v0, Landroid/view/Surface;

    invoke-virtual {p0}, Landroid/view/TextureView;->getSurfaceTexture()Landroid/graphics/SurfaceTexture;

    move-result-object v1

    invoke-direct {v0, v1}, Landroid/view/Surface;-><init>(Landroid/graphics/SurfaceTexture;)V

    iput-object v0, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    iget-object v1, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    invoke-virtual {v1, v0}, Lx/a;->q(Landroid/view/Surface;)V

    return-void

    :cond_1
    new-instance v0, Ljava/lang/IllegalStateException;

    const-string v1, "connectSurfaceToRenderer() should only be called when flutterRenderer and getSurfaceTexture() are non-null."

    invoke-direct {v0, v1}, Ljava/lang/IllegalStateException;-><init>(Ljava/lang/String;)V

    throw v0
.end method

.method private m()V
    .locals 2

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    if-eqz v0, :cond_1

    invoke-virtual {v0}, Lx/a;->r()V

    iget-object v0, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    if-eqz v0, :cond_0

    invoke-virtual {v0}, Landroid/view/Surface;->release()V

    const/4 v0, 0x0

    iput-object v0, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    :cond_0
    return-void

    :cond_1
    new-instance v0, Ljava/lang/IllegalStateException;

    const-string v1, "disconnectSurfaceFromRenderer() should only be called when flutterRenderer is non-null."

    invoke-direct {v0, v1}, Ljava/lang/IllegalStateException;-><init>(Ljava/lang/String;)V

    throw v0
.end method

.method private n()V
    .locals 1

    iget-object v0, p0, Lio/flutter/embedding/android/i;->e:Landroid/view/TextureView$SurfaceTextureListener;

    invoke-virtual {p0, v0}, Landroid/view/TextureView;->setSurfaceTextureListener(Landroid/view/TextureView$SurfaceTextureListener;)V

    return-void
.end method


# virtual methods
.method public a(Lx/a;)V
    .locals 2

    const-string v0, "FlutterTextureView"

    const-string v1, "Attaching to FlutterRenderer."

    invoke-static {v0, v1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v1, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    if-eqz v1, :cond_0

    const-string v1, "Already connected to a FlutterRenderer. Detaching from old one and attaching to new one."

    invoke-static {v0, v1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    iget-object v1, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    invoke-virtual {v1}, Lx/a;->r()V

    :cond_0
    iput-object p1, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    const/4 p1, 0x1

    iput-boolean p1, p0, Lio/flutter/embedding/android/i;->b:Z

    iget-boolean p1, p0, Lio/flutter/embedding/android/i;->a:Z

    if-eqz p1, :cond_1

    const-string p1, "Surface is available for rendering. Connecting FlutterRenderer to Android surface."

    invoke-static {v0, p1}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    invoke-direct {p0}, Lio/flutter/embedding/android/i;->l()V

    :cond_1
    return-void
.end method

.method public b()V
    .locals 2

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    if-eqz v0, :cond_0

    const/4 v0, 0x0

    iput-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    const/4 v0, 0x0

    iput-boolean v0, p0, Lio/flutter/embedding/android/i;->b:Z

    goto :goto_0

    :cond_0
    const-string v0, "FlutterTextureView"

    const-string v1, "pause() invoked when no FlutterRenderer was attached."

    invoke-static {v0, v1}, Lm/b;->f(Ljava/lang/String;Ljava/lang/String;)V

    :goto_0
    return-void
.end method

.method public c()V
    .locals 2

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    const-string v1, "FlutterTextureView"

    if-eqz v0, :cond_1

    invoke-virtual {p0}, Landroid/view/TextureView;->getWindowToken()Landroid/os/IBinder;

    move-result-object v0

    if-eqz v0, :cond_0

    const-string v0, "Disconnecting FlutterRenderer from Android surface."

    invoke-static {v1, v0}, Lm/b;->e(Ljava/lang/String;Ljava/lang/String;)V

    invoke-direct {p0}, Lio/flutter/embedding/android/i;->m()V

    :cond_0
    const/4 v0, 0x0

    iput-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    const/4 v0, 0x0

    iput-boolean v0, p0, Lio/flutter/embedding/android/i;->b:Z

    goto :goto_0

    :cond_1
    const-string v0, "detachFromRenderer() invoked when no FlutterRenderer was attached."

    invoke-static {v1, v0}, Lm/b;->f(Ljava/lang/String;Ljava/lang/String;)V

    :goto_0
    return-void
.end method

.method public getAttachedRenderer()Lx/a;
    .locals 1

    iget-object v0, p0, Lio/flutter/embedding/android/i;->c:Lx/a;

    return-object v0
.end method

.method public setRenderSurface(Landroid/view/Surface;)V
    .locals 0

    iput-object p1, p0, Lio/flutter/embedding/android/i;->d:Landroid/view/Surface;

    return-void
.end method
