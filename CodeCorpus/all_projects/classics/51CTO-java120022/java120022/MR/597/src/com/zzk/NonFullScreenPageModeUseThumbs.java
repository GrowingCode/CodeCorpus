package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Chapter;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfWriter;

public class NonFullScreenPageModeUseThumbs {
    
    public static void main(String args[]) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\全屏模式下显示缩略图.pdf"));// 关联文档对象与输出流
            writer.setViewerPreferences(PdfWriter.PageModeFullScreen);// 设置为全屏模式
            writer
                    .setViewerPreferences(PdfWriter.NonFullScreenPageModeUseThumbs);// 设置阅读器在全屏模式下显示缩略图
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light", "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体   
            Font fontChinese1 = new Font(Chinese, 18, Font.BOLDITALIC,BaseColor.RED);// 实例化字体类、设置字体大小和颜色
            Font fontChinese2 = new Font(Chinese, 15, Font.BOLDITALIC,BaseColor.BLUE);// 实例化字体类、设置字体大小和颜色
            Paragraph paragraph = new Paragraph("章节",fontChinese1);// 创建段落对象
            Chapter chapter = new Chapter(paragraph,1);// 创建章节对象
            paragraph = new Paragraph("小节一",fontChinese2);// 创建段落对象
            chapter.addSection(paragraph);// 添加小节
            paragraph = new Paragraph("小节二",fontChinese2);// 创建段落对象
            chapter.addSection(paragraph);// 添加小节
            document.add(chapter);// 向文档添加章节
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
