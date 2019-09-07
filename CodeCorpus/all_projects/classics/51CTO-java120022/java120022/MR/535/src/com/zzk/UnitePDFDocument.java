package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfCopy;
import com.itextpdf.text.pdf.PdfReader;

public class UnitePDFDocument {
    
    public static void main(String[] args) {
        String[] subFiles = { "c:\\原文档-1.pdf", "c:\\原文档-2.pdf", "c:\\原文档-3.pdf" }; // 待合并的PDF文档
        String newFile = "C:\\合并结果.pdf";// 合并后的新文档
        Document document = new Document();// 创建文本文档
        try {
            PdfCopy copy = new PdfCopy(document, new FileOutputStream(newFile));// 创建copy对象关联文档与输出流
            document.open();// 打开文档
            for (int i = 0; i < subFiles.length; i++) {// 做循环 获取待合并文件长度
                PdfReader reader = new PdfReader(subFiles[i]);// 读取待合并文件长度
                int totalPages = reader.getNumberOfPages();// 获得每个子文档的总页数
                for (int p = 1; p <= totalPages; p++) {// 遍历子文档的每一页
                    copy.addPage(copy.getImportedPage(reader, p));// 将子文档的每一页都添加到新文档中
                }
            }
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
