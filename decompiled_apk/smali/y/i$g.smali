.class public final enum Ly/i$g;
.super Ljava/lang/Enum;
.source ""


# annotations
.annotation system Ldalvik/annotation/EnclosingClass;
    value = Ly/i;
.end annotation

.annotation system Ldalvik/annotation/InnerClass;
    accessFlags = 0x4019
    name = "g"
.end annotation

.annotation system Ldalvik/annotation/Signature;
    value = {
        "Ljava/lang/Enum<",
        "Ly/i$g;",
        ">;"
    }
.end annotation


# static fields
.field public static final enum b:Ly/i$g;

.field public static final enum c:Ly/i$g;

.field public static final enum d:Ly/i$g;

.field public static final enum e:Ly/i$g;

.field public static final enum f:Ly/i$g;

.field private static final synthetic g:[Ly/i$g;


# instance fields
.field private final a:Ljava/lang/String;


# direct methods
.method static constructor <clinit>()V
    .locals 12

    new-instance v0, Ly/i$g;

    const-string v1, "STANDARD"

    const/4 v2, 0x0

    const/4 v3, 0x0

    invoke-direct {v0, v1, v2, v3}, Ly/i$g;-><init>(Ljava/lang/String;ILjava/lang/String;)V

    sput-object v0, Ly/i$g;->b:Ly/i$g;

    new-instance v1, Ly/i$g;

    const-string v3, "LIGHT_IMPACT"

    const/4 v4, 0x1

    const-string v5, "HapticFeedbackType.lightImpact"

    invoke-direct {v1, v3, v4, v5}, Ly/i$g;-><init>(Ljava/lang/String;ILjava/lang/String;)V

    sput-object v1, Ly/i$g;->c:Ly/i$g;

    new-instance v3, Ly/i$g;

    const-string v5, "MEDIUM_IMPACT"

    const/4 v6, 0x2

    const-string v7, "HapticFeedbackType.mediumImpact"

    invoke-direct {v3, v5, v6, v7}, Ly/i$g;-><init>(Ljava/lang/String;ILjava/lang/String;)V

    sput-object v3, Ly/i$g;->d:Ly/i$g;

    new-instance v5, Ly/i$g;

    const-string v7, "HEAVY_IMPACT"

    const/4 v8, 0x3

    const-string v9, "HapticFeedbackType.heavyImpact"

    invoke-direct {v5, v7, v8, v9}, Ly/i$g;-><init>(Ljava/lang/String;ILjava/lang/String;)V

    sput-object v5, Ly/i$g;->e:Ly/i$g;

    new-instance v7, Ly/i$g;

    const-string v9, "SELECTION_CLICK"

    const/4 v10, 0x4

    const-string v11, "HapticFeedbackType.selectionClick"

    invoke-direct {v7, v9, v10, v11}, Ly/i$g;-><init>(Ljava/lang/String;ILjava/lang/String;)V

    sput-object v7, Ly/i$g;->f:Ly/i$g;

    const/4 v9, 0x5

    new-array v9, v9, [Ly/i$g;

    aput-object v0, v9, v2

    aput-object v1, v9, v4

    aput-object v3, v9, v6

    aput-object v5, v9, v8

    aput-object v7, v9, v10

    sput-object v9, Ly/i$g;->g:[Ly/i$g;

    return-void
.end method

.method private constructor <init>(Ljava/lang/String;ILjava/lang/String;)V
    .locals 0
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "(",
            "Ljava/lang/String;",
            ")V"
        }
    .end annotation

    invoke-direct {p0, p1, p2}, Ljava/lang/Enum;-><init>(Ljava/lang/String;I)V

    iput-object p3, p0, Ly/i$g;->a:Ljava/lang/String;

    return-void
.end method

.method static a(Ljava/lang/String;)Ly/i$g;
    .locals 5

    invoke-static {}, Ly/i$g;->values()[Ly/i$g;

    move-result-object v0

    array-length v1, v0

    const/4 v2, 0x0

    :goto_0
    if-ge v2, v1, :cond_3

    aget-object v3, v0, v2

    iget-object v4, v3, Ly/i$g;->a:Ljava/lang/String;

    if-nez v4, :cond_0

    if-eqz p0, :cond_1

    :cond_0
    if-eqz v4, :cond_2

    invoke-virtual {v4, p0}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v4

    if-eqz v4, :cond_2

    :cond_1
    return-object v3

    :cond_2
    add-int/lit8 v2, v2, 0x1

    goto :goto_0

    :cond_3
    new-instance v0, Ljava/lang/NoSuchFieldException;

    new-instance v1, Ljava/lang/StringBuilder;

    invoke-direct {v1}, Ljava/lang/StringBuilder;-><init>()V

    const-string v2, "No such HapticFeedbackType: "

    invoke-virtual {v1, v2}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    invoke-virtual {v1, p0}, Ljava/lang/StringBuilder;->append(Ljava/lang/String;)Ljava/lang/StringBuilder;

    invoke-virtual {v1}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object p0

    invoke-direct {v0, p0}, Ljava/lang/NoSuchFieldException;-><init>(Ljava/lang/String;)V

    throw v0
.end method

.method public static valueOf(Ljava/lang/String;)Ly/i$g;
    .locals 1

    const-class v0, Ly/i$g;

    invoke-static {v0, p0}, Ljava/lang/Enum;->valueOf(Ljava/lang/Class;Ljava/lang/String;)Ljava/lang/Enum;

    move-result-object p0

    check-cast p0, Ly/i$g;

    return-object p0
.end method

.method public static values()[Ly/i$g;
    .locals 1

    sget-object v0, Ly/i$g;->g:[Ly/i$g;

    invoke-virtual {v0}, [Ly/i$g;->clone()Ljava/lang/Object;

    move-result-object v0

    check-cast v0, [Ly/i$g;

    return-object v0
.end method
