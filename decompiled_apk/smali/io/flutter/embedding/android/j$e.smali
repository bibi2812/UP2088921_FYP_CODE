.class final enum Lio/flutter/embedding/android/j$e;
.super Ljava/lang/Enum;
.source ""


# annotations
.annotation system Ldalvik/annotation/EnclosingClass;
    value = Lio/flutter/embedding/android/j;
.end annotation

.annotation system Ldalvik/annotation/InnerClass;
    accessFlags = 0x401a
    name = "e"
.end annotation

.annotation system Ldalvik/annotation/Signature;
    value = {
        "Ljava/lang/Enum<",
        "Lio/flutter/embedding/android/j$e;",
        ">;"
    }
.end annotation


# static fields
.field public static final enum a:Lio/flutter/embedding/android/j$e;

.field public static final enum b:Lio/flutter/embedding/android/j$e;

.field public static final enum c:Lio/flutter/embedding/android/j$e;

.field public static final enum d:Lio/flutter/embedding/android/j$e;

.field private static final synthetic e:[Lio/flutter/embedding/android/j$e;


# direct methods
.method static constructor <clinit>()V
    .locals 9

    new-instance v0, Lio/flutter/embedding/android/j$e;

    const-string v1, "NONE"

    const/4 v2, 0x0

    invoke-direct {v0, v1, v2}, Lio/flutter/embedding/android/j$e;-><init>(Ljava/lang/String;I)V

    sput-object v0, Lio/flutter/embedding/android/j$e;->a:Lio/flutter/embedding/android/j$e;

    new-instance v1, Lio/flutter/embedding/android/j$e;

    const-string v3, "LEFT"

    const/4 v4, 0x1

    invoke-direct {v1, v3, v4}, Lio/flutter/embedding/android/j$e;-><init>(Ljava/lang/String;I)V

    sput-object v1, Lio/flutter/embedding/android/j$e;->b:Lio/flutter/embedding/android/j$e;

    new-instance v3, Lio/flutter/embedding/android/j$e;

    const-string v5, "RIGHT"

    const/4 v6, 0x2

    invoke-direct {v3, v5, v6}, Lio/flutter/embedding/android/j$e;-><init>(Ljava/lang/String;I)V

    sput-object v3, Lio/flutter/embedding/android/j$e;->c:Lio/flutter/embedding/android/j$e;

    new-instance v5, Lio/flutter/embedding/android/j$e;

    const-string v7, "BOTH"

    const/4 v8, 0x3

    invoke-direct {v5, v7, v8}, Lio/flutter/embedding/android/j$e;-><init>(Ljava/lang/String;I)V

    sput-object v5, Lio/flutter/embedding/android/j$e;->d:Lio/flutter/embedding/android/j$e;

    const/4 v7, 0x4

    new-array v7, v7, [Lio/flutter/embedding/android/j$e;

    aput-object v0, v7, v2

    aput-object v1, v7, v4

    aput-object v3, v7, v6

    aput-object v5, v7, v8

    sput-object v7, Lio/flutter/embedding/android/j$e;->e:[Lio/flutter/embedding/android/j$e;

    return-void
.end method

.method private constructor <init>(Ljava/lang/String;I)V
    .locals 0
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "()V"
        }
    .end annotation

    invoke-direct {p0, p1, p2}, Ljava/lang/Enum;-><init>(Ljava/lang/String;I)V

    return-void
.end method

.method public static valueOf(Ljava/lang/String;)Lio/flutter/embedding/android/j$e;
    .locals 1

    const-class v0, Lio/flutter/embedding/android/j$e;

    invoke-static {v0, p0}, Ljava/lang/Enum;->valueOf(Ljava/lang/Class;Ljava/lang/String;)Ljava/lang/Enum;

    move-result-object p0

    check-cast p0, Lio/flutter/embedding/android/j$e;

    return-object p0
.end method

.method public static values()[Lio/flutter/embedding/android/j$e;
    .locals 1

    sget-object v0, Lio/flutter/embedding/android/j$e;->e:[Lio/flutter/embedding/android/j$e;

    invoke-virtual {v0}, [Lio/flutter/embedding/android/j$e;->clone()Ljava/lang/Object;

    move-result-object v0

    check-cast v0, [Lio/flutter/embedding/android/j$e;

    return-object v0
.end method
