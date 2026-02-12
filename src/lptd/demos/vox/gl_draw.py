from __future__ import annotations

from OpenGL.GL import (
    GL_COLOR_ATTACHMENT0,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LESS,
    GL_MODELVIEW,
    GL_RENDERBUFFER,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    GL_NEAREST,
    glBegin,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glClear,
    glClearDepth,
    glDeleteFramebuffers,
    glDeleteRenderbuffers,
    glDeleteTextures,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glColor3ub,
    glFramebufferTexture2D,
    glFramebufferRenderbuffer,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glRenderbufferStorage,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glViewport,
)

from . import config

_fbo: int | None = None
_fbo_tex: int | None = None
_fbo_depth: int | None = None
_fbo_w = 0
_fbo_h = 0


def setup_ortho(render_w: int, render_h: int) -> None:
    _setup_ortho_with_viewport(config.WIDTH, config.HEIGHT, render_w, render_h, enable_depth=False)


def _setup_ortho_with_viewport(
    view_w: int, view_h: int, proj_w: int, proj_h: int, *, enable_depth: bool
) -> None:
    glViewport(0, 0, view_w, view_h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Use a deterministic [0..1] depth range for screen-space primitives.
    glOrtho(0, proj_w, proj_h, 0, 0.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    if enable_depth:
        glEnable(GL_DEPTH_TEST)
        glDepthMask(True)
        glDepthFunc(GL_LESS)
    else:
        glDisable(GL_DEPTH_TEST)


def init_lowres_target(render_w: int, render_h: int) -> None:
    global _fbo, _fbo_tex, _fbo_depth, _fbo_w, _fbo_h
    if _fbo is not None and _fbo_w == render_w and _fbo_h == render_h:
        return

    if _fbo is not None:
        glDeleteFramebuffers(1, [_fbo])
        _fbo = None
    if _fbo_tex is not None:
        glDeleteTextures([_fbo_tex])
        _fbo_tex = None
    if _fbo_depth is not None:
        glDeleteRenderbuffers(1, [_fbo_depth])
        _fbo_depth = None

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        render_w,
        render_h,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        None,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)

    depth_rb = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, render_w, render_h)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Framebuffer incomplete: 0x{status:04x}")

    _fbo = fbo
    _fbo_tex = tex
    _fbo_depth = depth_rb
    _fbo_w = render_w
    _fbo_h = render_h


def begin_lowres_pass(render_w: int, render_h: int) -> None:
    if _fbo is None:
        init_lowres_target(render_w, render_h)
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo)
    _setup_ortho_with_viewport(render_w, render_h, render_w, render_h, enable_depth=True)
    glClearDepth(1.0)
    glClear(GL_DEPTH_BUFFER_BIT)


def blit_lowres_to_screen(window_w: int, window_h: int) -> None:
    if _fbo_tex is None:
        return
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    _setup_ortho_with_viewport(window_w, window_h, window_w, window_h, enable_depth=False)
    glEnable(GL_TEXTURE_2D)
    # Fixed-function texturing modulates by current color; force neutral white.
    glColor3ub(255, 255, 255)
    glBindTexture(GL_TEXTURE_2D, _fbo_tex)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(window_w, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(window_w, window_h)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, window_h)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
