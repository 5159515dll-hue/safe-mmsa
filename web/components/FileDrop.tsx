"use client";

import { useRef, useState } from "react";

// 视频 / 音频上传拖放区：虚线边框，悬浮/拖拽时变蓝，选中后显示文件名 + 清除。
export function FileDrop({
  label,
  accept,
  file,
  onFile,
  hint,
  code,
}: {
  label: string;
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
  hint?: string;
  code?: string;
}) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <label className="filedrop">
      <span className="lbl">{code && <em className="chcode">{code}</em>}{label}</span>
      <div
        className={"filedrop-zone" + (drag ? " drag" : "")}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          onFile(e.dataTransfer.files?.[0] ?? null);
        }}
      >
        <span className={"filedrop-name" + (file ? " has" : "")}>
          {file ? file.name : hint ?? "点击或拖拽文件到此处"}
        </span>
        {file && (
          <span
            className="filedrop-clear"
            onClick={(e) => {
              e.stopPropagation();
              onFile(null);
              if (inputRef.current) inputRef.current.value = "";
            }}
          >
            清除
          </span>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          style={{ display: "none" }}
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
      </div>
    </label>
  );
}
