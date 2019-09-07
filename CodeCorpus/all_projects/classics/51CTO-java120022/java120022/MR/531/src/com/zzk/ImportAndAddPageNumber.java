package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;
import com.itextpdf.text.pdf.PdfWriter;

public class ImportAndAddPageNumber {
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream("c:\\页码.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            document.add(new Paragraph("No. 1 page")); // 向文档中添加内容
            document.newPage();
            document.add(new Paragraph("No. 2 page")); // 向文档中添加内容
            document.close();// 关闭文档对象
            PdfReader reader = new PdfReader("c:\\页码.pdf");// 创建“页码.pdf”的PdfReader对象
            int totalPages = reader.getNumberOfPages();
            PdfStamper stamp = new PdfStamper(reader, new FileOutputStream(
                    "c:\\导入并添加页码.pdf"));// 创建PdfStamper对象
            BaseFont chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义字体
            PdfContentByte under = null;
            for (int i = 1; i <= totalPages; i++) {
                under = stamp.getUnderContent(i);// 获得每一页的内容
                under.beginText();// 标记文本开始
                under.setFontAndSize(chinese, 18);// 设置字体和字号
                under.setTextMatrix(280, 15);// 设置页码的显示位置
                under.showText("第" + i + "页");// 添加页脚
                under.endText();// 标记文本结束
            }
            stamp.close();// PdfStamper对象，将从“页码.pdf”中读取的文档添加页码后写入“添加页码.pdf”
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
