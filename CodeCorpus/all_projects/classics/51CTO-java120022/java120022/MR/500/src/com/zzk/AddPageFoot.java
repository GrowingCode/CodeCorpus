package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.PageSize;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;
import com.itextpdf.text.pdf.PdfWriter;

public class AddPageFoot {
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\页眉页脚.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			document.add(new Paragraph("Add Page Top And Foot.")); // 向文档中添加内容
			document.close();// 关闭文档对象
			PdfReader reader = new PdfReader("c:\\页眉页脚.pdf");// 创建“页眉页脚.pdf”的PdfReader对象
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream("c:\\添加页眉页脚.pdf"));// 创建PdfStamper对象
			PdfContentByte over = stamp.getOverContent(1);// 获得第一页的内容
			over.setTextRise(810);// 文本上移到810的位置
			over.beginText();// 标记文本开始
			BaseFont chinese = BaseFont.createFont("STSong-Light", "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义字体
			over.setFontAndSize(chinese, 18);// 设置字体和字号
			over.showText("                                           页眉的内容");// 添加页眉
			over.endText();// 标记文本结束
			stamp.insertPage(2, PageSize.A4);// 增加新的一页，为新页添加页脚
			PdfContentByte under = stamp.getUnderContent(2);// 获得第二页的内容
			under.setTextRise(15);// 文本上移到15的位置
			under.beginText();// 标记文本开始
			under.setFontAndSize(chinese, 18);// 设置字体和字号
			under.showText("                                          页脚的内容");// 添加页脚
			under.endText();// 标记文本结束
			stamp.close();// PdfStamper对象，将从“页眉页脚.pdf”中读取的文档添加页眉页脚后写入“添加页眉页脚.pdf”
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
